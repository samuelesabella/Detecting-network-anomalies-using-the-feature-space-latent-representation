from collections import defaultdict
import itertools
from skorch.net import NeuralNet
from skorch.dataset import Dataset
import pandas as pd
from scipy.stats import truncnorm
from sklearn import preprocessing
from sklearn.manifold import TSNE
from skorch.callbacks import EpochScoring
from tqdm import tqdm
import matplotlib.pyplot as plt
import more_itertools as mit
import numpy as np
import skorch
import torch

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)


NORMAL_TRAFFIC = np.array([ 0. ])
ATTACK_TRAFFIC = np.array([ 1. ]) 


class ContextCriterion():
    def score(self, model_output, target):
        raise NotImplementedError

    def __call__(self, p1, p2, p3=None):
        if isinstance(p1, NeuralNet): # p1=Model, p2=dataset.X, p3=dataset.y
            with torch.no_grad():
                mout = p1.forward(p2)
                return self.score(mout, p3)
        else: # p1=model_output, p2=dataset.y
            return self.score(p1, p2) 


class WindowedDataGenerator():
    def __init__(self, overlapping, context_len):
        self.overlapping = overlapping
        self.context_len = context_len
        self.activity_len = int(context_len / 2)
        self.window_stepsize = max(int(context_len * (1 - overlapping)), 1) 

    def dataframe_windows(self, df):
        df_len = len(df)
        wnds = mit.windowed(range(df_len), self.context_len, step=self.window_stepsize)
        wnds = filter(lambda x: None not in x, wnds)
        wnds_values = map(lambda x: df.iloc[list(x)].reset_index(), wnds)
        return wnds_values
    
    def anomaly_metadata(self, context):
        anomaly_perc = len(context[context["_isanomaly"] != "none"]) / self.context_len
        anomaly_type = "none"
        if anomaly_perc > 0:
            anomaly_type = context.loc[context["_isanomaly"] != "none", "_isanomaly"].iloc[0]
        return anomaly_type, anomaly_perc

    def generate_context(self, df: pd.DataFrame):
        samples = defaultdict(list)
        channels = [c for c in df.columns if c[0] != "_"]
    
        logging.debug("Windowing time series for each host")
        host_ts = df.groupby(level=["_host"])
        for host, ts in tqdm(host_ts):
            windows = self.dataframe_windows(ts)
            for context in windows:
                anomaly_type, anomaly_perc = self.anomaly_metadata(context)
                samples["anomaly_type"].append(anomaly_type)
                samples["anomaly_perc"].append(anomaly_perc)

                ctxvalues = context[channels].values
                samples["context"].append(ctxvalues)
                samples["host"].append(host)
        samples = { k: np.stack(v) for k, v in samples.items() }
        return samples

    @staticmethod
    def alternate_merge(ll):
        return list(itertools.chain(*zip(*ll)))
    
    def sk_dataset(self, context_dictionary):
        skdset = {}
        skdset["context"] = torch.Tensor(context_dictionary["context"])
        # Host to id
        skdset["host"] = preprocessing.LabelEncoder().fit_transform(context_dictionary["host"])
        # context_dictionary["_device_category"] = preprocessing.LabelEncoder().fit_transform(context_dictionary["_device_category"])
        an_perc = context_dictionary["anomaly_perc"]
        Y = np.where(an_perc==0, NORMAL_TRAFFIC, an_perc)
        Y = np.where(Y > 0, ATTACK_TRAFFIC, Y)
        Y = torch.Tensor(Y)
        
        return self.Dataset2GPU(Dataset(skdset, Y))

    def Dataset2GPU(self, dataset):
        if torch.cuda.is_available():
            dataset.X["context"] = dataset.X["context"].cuda()
            dataset.y = dataset.y.cuda()
        return dataset

    def __call__(self, df_collection, to_sk_dataset=True, filter_anomaly=True):
        model_input = defaultdict(list)

        for df in df_collection:
            ctxs = self.generate_context(df)
            for k, v in ctxs.items():
                model_input[k].append(v)
        model_input = { k: np.array(self.alternate_merge(v)) for k, v in model_input.items() }
        
        if filter_anomaly:
            normal_mask = np.where(model_input["anomaly_perc"] == 0)[0]
            model_input = { k: x[normal_mask] for k, x in model_input.items() }

        if to_sk_dataset:
            return self.sk_dataset(model_input)
        return model_input


class WindowedAnomalyDetector(torch.nn.Module):
    def __init__(self, wd: WindowedDataGenerator=None):
        super(WindowedAnomalyDetector, self).__init__()
        if wd is None:
            print("WindowedDataGenerator not configured, anomaly detector not configured for pointwise prection")
        else:
            self.pointwise_ctxs = WindowedDataGenerator(1., wd.context_len)

    def toembedding(self, x):
        raise NotImplementedError()
    
    def context_anomaly(self, ctx):
        raise NotImplementedError()

    def pointwise_prediction(self, samples):
        if not isinstance(samples, list):
            samples = [samples]

        channels = [c for c in samples[0].columns if c[0] != "_"]
        for df in samples:
            host_ts = df.groupby(level=["_host"])
            for _, host_df in tqdm(host_ts):
                y_hat = []
                windows = self.pointwise_ctxs.dataframe_windows(host_df)
                for context in windows:
                    ctx = context[channels].values
                    yi_hat = self.context_anomaly(ctx)
                    y_hat.append(yi_hat)
                host_df["y_hat"] = np.concat(y_hat)

