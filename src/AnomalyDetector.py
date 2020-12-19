from collections import defaultdict
import itertools
from skorch.net import NeuralNet
from skorch.dataset import Dataset
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import more_itertools as mit
import numpy as np
import skorch
import torch

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)


NORMAL_TRAFFIC = 0.
ATTACK_TRAFFIC = 1.


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
        skdset["host"] = preprocessing.LabelEncoder().fit_transform(context_dictionary["host"])
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


class WindowedAnomalyDetector(skorch.net.NeuralNet):
    def __init__(self, *args, context_len=None, **kwargs):
        self.pointwise_ctxs = None
        if context_len is not None:
            self.initialize_context(context_len)
        super(WindowedAnomalyDetector, self).__init__(*args, **kwargs)
    
    def initialize_context(self, context_len):
        self.context_len = context_len
        self.pointwise_ctxs = WindowedDataGenerator(1., context_len)

    def fit(self, *args, **kwargs):
        context_len = args[0].X["context"].size(1)
        self.initialize_context(context_len)
        super().fit(*args, **kwargs)

    def pointwise_embedding(self, samples):
        activity_len = int(self.context_len / 2)
        extract_activity = (lambda ctx: self.module_.toembedding(ctx[:, :activity_len]))

        return self.pointwise(samples, self.module_.toembedding, "_embedding", pad_with=np.nan)
    
    def pointwise_anomaly(self, samples, one_hot=False):
        return self.pointwise(samples, self.module_.context_anomaly, "_y_hat")

    def pointwise(self, samples, fun, label, pad_with=0.):
        if self.pointwise_ctxs is None: 
            raise AttributeError("Not fitted, missing context len") 

        if not isinstance(samples, list):
            samples = [samples]

        activity_len = int(self.context_len / 2)
        res = [ [] for i in range(len(samples)) ]
        channels = [c for c in samples[0].columns if c[0] != "_"]
        for i, df in enumerate(samples):
            host_ts = df.groupby(level=["_host"])
            for _, host_df in host_ts:
                windows = self.pointwise_ctxs.dataframe_windows(host_df)
                ctx_batch = np.stack([ ctx[channels].values for ctx in windows ])

                def aperc(ctx):
                    return self.pointwise_ctxs.anomaly_metadata(ctx)[1] 
                windows = self.pointwise_ctxs.dataframe_windows(host_df)
                aperc = np.array([ aperc(ctx) for ctx in windows ])
                vaperc = np.full((len(host_df), 1), pad_with).squeeze()
                vaperc[activity_len:-activity_len+1] = aperc
                host_df["_aperc"] = vaperc

                with torch.no_grad():
                    pred = fun(torch.tensor(ctx_batch))

                # Fix windowing padding with zeros (hope no anomaly)
                if len(pred.shape) == 1:
                    y_hat = np.full((len(host_df), 1), pad_with).squeeze()
                    y_hat[activity_len:-activity_len+1] = pred.numpy()
                else:
                    y_hat = np.full((len(host_df), pred.size(1)), pad_with)
                    y_hat[activity_len:-activity_len+1] = pred.numpy()
                    y_hat = [ np.nan if np.isnan(x).any() else x for x in list(y_hat) ]
                
                host_df[label] = y_hat
                res[i].append(host_df)
            res[i] = pd.concat(res[i])

        if not isinstance(samples, list):
            return res[0]
        return res

