from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import make_scorer
from sklearn.model_selection import ParameterGrid
from pathlib import Path
from skorch.callbacks import EpochScoring
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skmetrics 
from sklearn.model_selection import KFold
from skorch.net import NeuralNet
import argparse
import data_generator as generator
import logging
import model_codebase as cb
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch


def tqdm_iterator(dataset, **kwargs):
    return tqdm(torch.utils.data.DataLoader(dataset, **kwargs))


# ----- ----- PREPROCESSING ----- ----- #
# ----- ----- ------------- ----- ----- #
ATTACKS = [
    # Tuesday
    ["Brute Force",  "FTP-Patator",      "2017-07-04T09:20:00", "2017-07-04T10:20:00", "192.168.10.50"],
    ["Brute Force",  "SSH-Patator",      "2017-07-04T14:00:00", "2017-07-04T15:00:00", "192.168.10.50"],
    # Wednesday
    ["DoS",          "slowloris",        "2017-07-05T09:47:00", "2017-07-05T10:10:00", "192.168.10.50"],
    ["DoS",          "Slowhttptest",     "2017-07-05T10:14:00", "2017-07-05T10:35:00", "192.168.10.50"],
    ["DoS",          "Hulk",             "2017-07-05T10:43:00", "2017-07-05T11:00:00", "192.168.10.50"],         
    ["DoS",          "GoldenEye",        "2017-07-05T11:10:00", "2017-07-05T11:23:00", "192.168.10.50"],         
    ["Side_channel", "Heartbleed",       "2017-07-05T15:12:00", "2017-07-05T15:32:00", "192.168.10.51"],
    # Thursday
    ["Web Attack",    "Brute Force",     "2017-07-06T9:20:00",  "2017-07-06T10:00:00", "192.168.10.50"],
    ["Web Attack",    "XSS",             "2017-07-06T10:15:00", "2017-07-06T10:35:00", "192.168.10.50"],
    ["Web Attack",    "Sql Injection",   "2017-07-06T10:40:00", "2017-07-06T10:42:00", "192.168.10.50"],
    ["Infiltration",  "Meta exploit",    "2017-07-06T14:19:00", "2017-07-06T14:21:00", "192.168.10.8"],
    ["Infiltration",  "Meta exploit",    "2017-07-06T14:33:00", "2017-07-06T14:35:00", "192.168.10.8"],
    ["Infiltration",  "Cool disk",       "2017-07-06T14:53:00", "2017-07-06T15:00:00", "192.168.10.25"],         
    ["Infiltration",  "Nmap - Portscan", "2017-07-06T15:04:00", "2017-07-06T15:45:00", "*"],
    # Friday
    ["Botnet",        "ARES",            "2017-07-07T10:02:00", "2017-07-07T11:02:00", "192.168.10.15"],
    ["Botnet",        "ARES",            "2017-07-07T10:02:00", "2017-07-07T11:02:00", "192.168.10.9"],
    ["Botnet",        "ARES",            "2017-07-07T10:02:00", "2017-07-07T11:02:00", "192.168.10.14"],
    ["Botnet",        "ARES",            "2017-07-07T10:02:00", "2017-07-07T11:02:00", "192.168.10.5"],
    ["Botnet",        "ARES",            "2017-07-07T10:02:00", "2017-07-07T11:02:00", "192.168.10.8"],
    ["DDoS",          "LOIT",            "2017-07-07T15:56:00", "2017-07-07T16:16:00", "192.168.10.50"],
    ["Port Scan",     "firewall on",     "2017-07-07T13:55:00", "2017-07-07T13:57:00", "192.168.10.50"],
    ["Port Scan",     "firewall on",     "2017-07-07T13:58:00", "2017-07-07T14:00:00", "192.168.10.50"],
    ["Port Scan",     "firewall on",     "2017-07-07T14:01:00", "2017-07-07T14:04:00", "192.168.10.50"],
    ["Port Scan",     "firewall on",     "2017-07-07T14:05:00", "2017-07-07T14:07:00", "192.168.10.50"],
    ["Port Scan",     "firewall on",     "2017-07-07T14:08:00", "2017-07-07T14:10:00", "192.168.10.50"],
    ["Port Scan",     "firewall on",     "2017-07-07T14:11:00", "2017-07-07T14:13:00", "192.168.10.50"],
    ["Port Scan",     "firewall on",     "2017-07-07T14:14:00", "2017-07-07T14:16:00", "192.168.10.50"],
    ["Port Scan",     "firewall on",     "2017-07-07T14:17:00", "2017-07-07T14:19:00", "192.168.10.50"],
    ["Port Scan",     "firewall on",     "2017-07-07T14:20:00", "2017-07-07T14:21:00", "192.168.10.50"],
    ["Port Scan",     "firewall on",     "2017-07-07T14:22:00", "2017-07-07T14:24:00", "192.168.10.50"],
    ["Port Scan",     "firewall on",     "2017-07-07T14:33:00", "2017-07-07T14:33:00", "192.168.10.50"],
    ["Port Scan",     "firewall on",     "2017-07-07T14:35:00", "2017-07-07T14:35:00", "192.168.10.50"],
    ["Port Scan",     "NMap sS",         "2017-07-07T14:51:00", "2017-07-07T14:53:00", "192.168.10.50"],
    ["Port Scan",     "NMap sT",         "2017-07-07T14:54:00", "2017-07-07T14:56:00", "192.168.10.50"],
    ["Port Scan",     "NMap sF",         "2017-07-07T14:57:00", "2017-07-07T14:59:00", "192.168.10.50"],
    ["Port Scan",     "NMap sX",         "2017-07-07T15:00:00", "2017-07-07T15:02:00", "192.168.10.50"],
    ["Port Scan",     "NMap sN",         "2017-07-07T15:03:00", "2017-07-07T15:05:00", "192.168.10.50"],
    ["Port Scan",     "NMap sP",         "2017-07-07T15:06:00", "2017-07-07T15:07:00", "192.168.10.50"],
    ["Port Scan",     "NMap sV",         "2017-07-07T15:08:00", "2017-07-07T15:10:00", "192.168.10.50"],
    ["Port Scan",     "NMap sU",         "2017-07-07T15:11:00", "2017-07-07T15:12:00", "192.168.10.50"],
    ["Port Scan",     "NMap sO",         "2017-07-07T15:13:00", "2017-07-07T15:15:00", "192.168.10.50"],
    ["Port Scan",     "NMap sA",         "2017-07-07T15:16:00", "2017-07-07T15:18:00", "192.168.10.50"],
    ["Port Scan",     "NMap sW",         "2017-07-07T15:19:00", "2017-07-07T15:21:00", "192.168.10.50"],
    ["Port Scan",     "NMap sR",         "2017-07-07T15:22:00", "2017-07-07T15:24:00", "192.168.10.50"],
    ["Port Scan",     "NMap sL",         "2017-07-07T15:25:00", "2017-07-07T15:25:00", "192.168.10.50"],
    ["Port Scan",     "NMap sI",         "2017-07-07T15:26:00", "2017-07-07T15:27:00", "192.168.10.50"],
    ["Port Scan",     "NMap b" ,         "2017-07-07T15:28:00", "2017-07-07T15:29:00", "192.168.10.50"]
]


class Cicids2017Preprocessor(generator.Preprocessor):
    @staticmethod
    def label(df):                                          
        def to_dt(x):
            return pd.to_datetime(x, format='%Y-%m-%dT%H:%M:%S.%f')
        
        df["attack"] = "none"
        host_idxs = df.index.get_level_values("host")
        times_idxs = df.index.get_level_values("_time")
        
        for atype, adetail, start, stop, host in ATTACKS:
            dt_start = to_dt(start)
            dt_stop = to_dt(stop)
            if host == "*":
                host_selector = True
            else:
                host_selector = (host_idxs==host)
            time_selector = (times_idxs > dt_start) & (times_idxs < dt_stop)
            df.loc[host_selector & time_selector, "attack"] = f"{atype}__{adetail}"
        return df
    
    def preprocessing(self, df, **kwargs):
        # Filtering hosts ..... #
        df = df[df.index.get_level_values('host').str.contains("192.168.10.")]
        # Passing the ball ..... #
        preproc_df = super().preprocessing(df, **kwargs)
        # Removing initial non zero traffic ..... #
        index_hours = preproc_df.index.get_level_values("_time").hour
        working_hours = (index_hours > 8) & (index_hours < 17)
        preproc_df = preproc_df[working_hours]
    
        return Cicids2017Preprocessor.label(preproc_df)


# ----- ----- DATA GENERATOR ----- ----- #
# ----- ----- -------------- ----- ----- #
CICIDS2017_IPV4_NETMAP = defaultdict(lambda : "unknown device class", {
    "192.168.10.3": "server",
    "192.168.10.50": "server",
    "192.168.10.51": "server",
    "205.174.165.68": "server",
    "205.174.165.66": "server", 

    "192.168.10.19": "pc",
    "192.168.10.17": "pc",
    "192.168.10.16": "pc",
    "192.168.10.12": "pc",
    "192.168.10.9": "pc",
    "192.168.10.5": "pc",
    "192.168.10.8": "pc",
    "192.168.10.14": "pc",
    "192.168.10.15": "pc",
    "192.168.10.25": "pc",
})


class CICIDS2017(generator.FluxDataGenerator):
    def __init__(self, *args):
        super(CICIDS2017, self).__init__(*args, None)

    def category_map(self, new_samples):
        return new_samples['host'].map(CICIDS2017_IPV4_NETMAP)


# ----- ----- EXPERIMENTS ----- ----- #
# ----- ----- ----------- ----- ----- #

# Reproducibility .... #
SEED = 117697 
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def prepare_dataset(df):
    pr = Cicids2017Preprocessor()
    
    df_train = df[df.index.get_level_values("_time").day == 3]
    df_train = pr.preprocessing(df_train, update=True)
    train_set = cb.ts_windowing(df_train, overlapping=.8)
    return train_set, None

    test_set = defaultdict(list)
    for d in [4, 5]: #, 6, 7]:
        df_day = df[df.index.get_level_values("_time").day == d]
        df_day_preproc = pr.preprocessing(df_day, update=False)
        test_day = cb.ts_windowing(df_day_preproc)
        
        attacks_rows = torch.where(test_day["attack"] == cb.ATTACK_TRAFFIC)[0].numpy()
        normal_rows = torch.where(test_day["attack"] == cb.NORMAL_TRAFFIC)[0].numpy()
        normal_rows = np.random.choice(normal_rows, len(attacks_rows), replace=False)

        sampled_rows = np.concatenate([normal_rows, attacks_rows])
        sample_df = test_day["X"].loc[sampled_rows]
        sample_coh_label = test_day["coherency_label"][sampled_rows]
        sample_attack_label = test_day["attack"][sampled_rows]

        test_set["X"].append(sample_df)
        test_set["coherency_label"].append(sample_coh_label)
        test_set["attack"].append(sample_attack_label)
    
    # Resetting sample indexes for test set
    test_samples = []
    for day_df in test_set["X"]:
        samples = day_df.groupby(level="sample_idx", sort=False)
        test_samples.extend([sg[1].reset_index(level="sample_idx") for sg in samples])
    test_set_len = len(test_samples)
    test_set["X"] = pd.concat(test_samples, keys=range(test_set_len), names=["sample_idx"])
    test_set["X"].drop(columns=["sample_idx"], inplace=True)
    
    # concatenating labels 
    test_set["coherency_label"] = torch.cat(test_set["coherency_label"])
    test_set["attack"] = torch.cat(test_set["attack"])

    return train_set, test_set


def store_dataset(dataset, path):
    path.mkdir(parents=True, exist_ok=True)
    for k, v in dataset.items():
        if torch.is_tensor(v):
            torch.save(v, path / f"{k}.torch.pkl")
        elif isinstance(v, pd.DataFrame):
            pd.to_pickle(v, path / f"{k}.pandas.pkl") 


def load_dataset(path):
    dataset = {}
    for f in os.listdir(path):
        fpath = path / f
        key = f.split(".")[0]
        if "torch" in f:
            dataset[key] = torch.load(fpath)
        elif "pandas" in f:
            dataset[key] = pd.read_pickle(fpath)
    return dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description="mEmbedding training")
    parser.add_argument("--timeseries", "-t", help="Timeseries data path", default=None, type=Path)
    parser.add_argument("--outfile", "-o", help="Grid-search output file", type=Path)
    # parser.add_argument("--dataset", "-d", help="Training/testing dataset", default=None, type=Path)
    args = parser.parse_args()

    # Data preprocessing ..... #
    df = pd.read_pickle(args.timeseries)
    train_set, test_set = prepare_dataset(df)# if not args.dataset: 

    # df = pd.read_pickle(args.timeseries)
    #     train_set, test_set = prepare_dataset(df)
    #     store_dataset(train_set, args.timeseries.parent / "train") 
    #     store_dataset(test_set, args.timeseries.parent / "test")
    # else:
    #     train_set = load_dataset(args.dataset / "train")
    #     test_set = load_dataset(args.dataset / "test")
    # import pdb; pdb.set_trace() 
    
    X_train = cb.X2split_tensors(train_set["X"])
    Y_train = { k: train_set[k] for k in ["coherency_label", "attack"] }

    # Scoring ..... #
    coherence_accuracy = cb.Ts2VecScore(skmetrics.accuracy_score, "coherency_label")
    epoch_coh_acc = EpochScoring(coherence_accuracy, target_extractor=coherence_accuracy)
    
    coherence_rec = cb.Ts2VecScore(skmetrics.recall_score, "coherency_label")
    epoch_coh_rec = EpochScoring(coherence_rec, target_extractor=coherence_rec)

    coherence_precision = cb.Ts2VecScore(skmetrics.precision_score, "coherency_label")
    epoch_coh_prec = EpochScoring(coherence_precision, target_extractor=coherence_precision)

    # Grid hyperparams ..... #
    kf = KFold(n_splits=5)
    net = NeuralNet(
        cb.Ts2Vec, 
        cb.Contextual_Coherency,
        optimizer=torch.optim.Adam,
        callbacks=[
            ("coherency precision", epoch_coh_prec),
            ("coherency recall", epoch_coh_rec),
            ("coherency accuracy", epoch_coh_acc),
        ])    
    grid_params = ParameterGrid({
        'lr': [0.033, 0.02],
        'max_epochs': [1],
    })

    # Grid search ..... #
    logging.debug("Starting grid search")
    grid_res = pd.DataFrame()
    for params in tqdm(grid_params):  
        # Parameter initialization
        for k, v in params.items():
            setattr(net, k, v)
        
        # Kfold fitting 
        for train_index, vl_index in kf.split(Y_train["coherency_label"]):
            X_cv_train = { k: v[train_index, :] for k, v in X_train.items() }
            Y_cv_train = { k: v[train_index, :] for k, v in Y_train.items() }

            X_cv_vl = { k: v[vl_index, :] for k, v in X_train.items() }
            Y_cv_vl = { k: v[vl_index, :] for k, v in Y_train.items() }
            
            fmodel = net.fit(X_cv_train, Y_cv_train)
            # Storing fold results
            ignore_keys = ['batches', 'epoch', 'train_batch_count', 'valid_batch_count']
            kfold_res = {k:v for k,v in fmodel.history[-1].items() if k not in ignore_keys}
            kfold_res.update({ f"hyperparam_{k}": v for k, v in params.items() })
            grid_res = pd.concat([grid_res, pd.Series(kfold_res).to_frame().T])
    grid_res.to_pickle(args.outputfile)



