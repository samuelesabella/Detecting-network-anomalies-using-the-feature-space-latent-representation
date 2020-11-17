from collections import defaultdict
import os
import math
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from skorch.callbacks import EarlyStopping
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from skorch.net import NeuralNet
from tqdm import tqdm
import argparse
import data_generator as generator
from datetime import datetime
import logging
import gzip
import _pickle as cPickle
import model_codebase as cb
import numpy as np
import pandas as pd
import random
import sklearn.metrics as skmetrics 
import torch


# Reproducibility .... #
SEED = 11769
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# CONSTANTS ..... #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_OVERLAPPING = .95
PATIENCE = 25
FLEVEL = "MAGIK"

LOSS = cb.Contextual_Coherency


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
        
        df["isanomaly"] = "none"
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
            df.loc[host_selector & time_selector, "isanomaly"] = f"{atype}__{adetail}"
        return df
    
    def preprocessing(self, df, fit=False):
        # Filtering hosts ..... #
        df = df[df.index.get_level_values('host').str.contains("192.168.10.")]
        df = df.drop(["host_unreachable_flows:flows_as_client",
                      "dns_qry_sent_rsp_rcvd:replies_error_packets",
                      "dns_qry_rcvd_rsp_sent:replies_error_packets"], axis=1) # All zeros in the dataset
        # Removing initial non zero traffic ..... #
        index_hours = df.index.get_level_values("_time").hour
        working_hours = (index_hours > 8) & (index_hours < 17)
        df = df[working_hours]
        # Passing the ball ..... #
        days_df = []
        days = np.unique(df.index.get_level_values("_time").day)
        for d in days:
            d = df[df.index.get_level_values("_time").day == d]
            d = super().preprocessing(df, fit=False)
            days_df.append(d)
        preproc_df = pd.concat(days_df)
        if self.compute_discrtz: 
            preproc_df = self.discretize(preproc_df, fit)
    
        return Cicids2017Preprocessor.label(preproc_df)


# ----- ----- DATA GENERATOR ----- ----- #
# ----- ----- -------------- ----- ----- #
class CICIDS2017(generator.FluxDataGenerator):
    def __init__(self, *args):
        super(CICIDS2017, self).__init__(*args, None)

        self.device_map = defaultdict(lambda : "unknown device class", {
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

            "192.168.10.255": "broadcast",
        })

    def category_map(self, new_samples):
        return new_samples['host'].map(self.device_map)


# ----- ----- EXPERIMENTS ----- ----- #
# ----- ----- ----------- ----- ----- #
def history2dframe(net, labels=None):
    ignore_keys = ["batches", "train_batch_count", "valid_batch_count"]
    best_epoch = next(x for x in reversed(net.history) if x["valid_loss_best"])
    if labels:
        best_epoch.update({ f"hyperparam_{k}": v for k, v in labels.items() })

    s = pd.Series(best_epoch).to_frame().T
    s = s.drop(columns=ignore_keys)
    return s.infer_objects()


def ts2vec_cicids2017(train, validation, validation_attacks, outpath):
    dist_plot = cb.DistPlot(outpath)
    loss_plot = cb.EpochPlot(outpath, ["train_loss", "valid_loss"])
    rec_prec_plot = cb.EpochPlot(outpath, ["valid_precision_score", "valid_recall_score"])
    net = NeuralNet(
        cb.STC, LOSS, optimizer=torch.optim.Adam, 
        iterator_train__shuffle=True,
        lr=1e-4, batch_size=6090, max_epochs=4000,
        device=DEVICE, verbose=1, train_split=None,
        callbacks=[
            # *cb.Ts2VecScore(skmetrics.accuracy_score).epoch_score(),
            cb.Ts2VecScore(skmetrics.recall_score, data=validation_attacks).epoch_score(on_train=False),
            cb.Ts2VecScore(skmetrics.precision_score, data=validation_attacks).epoch_score(on_train=False),
            cb.Ts2VecScore(skmetrics.roc_auc_score, data=validation_attacks).epoch_score(on_train=False),
            dist_plot, loss_plot, rec_prec_plot,
            EarlyStopping("valid_loss", lower_is_better=True, patience=PATIENCE)
        ])    

    # Retrain on whole dataset ..... #
    net.train_split = predefined_split(validation)
    net.fit(train)
    
    return net.module_, history2dframe(net)


# ----- ----- GRID SEARCH ----- ----- #
# ----- ----- ----------- ----- ----- #
def setparams(net, params):
    for k, v in params.items():
        setattr(net, k, v)


def grid_search(train, valid, grid_params, outpath):
    grid_res = pd.DataFrame()
    
    dist_plot = cb.DistPlot(outpath)
    loss_plot = cb.EpochPlot(outpath, ["train_loss", "valid_loss"])
    net = NeuralNet(
        cb.STC, LOSS, 
        optimizer=torch.optim.Adam, device=DEVICE,
        iterator_train__shuffle=True,
        verbose=1, train_split=None,
        callbacks=[
            dist_plot, loss_plot,
            EarlyStopping("valid_loss", lower_is_better=True, patience=PATIENCE),
            cb.Ts2VecScore(skmetrics.accuracy_score).epoch_score(),
            cb.Ts2VecScore(skmetrics.recall_score).epoch_score(),
            cb.Ts2VecScore(skmetrics.precision_score).epoch_score(),
            cb.Ts2VecScore(skmetrics.roc_auc_score).epoch_score()
        ])    

    # Grid search ..... #
    logging.debug("Starting grid search")
    grid_pbar = tqdm(grid_params)
    for params in grid_pbar:  
        grid_pbar.set_description(str(params))
        setparams(net, params) 
        dist_plot.set_flabel(params)
        loss_plot.set_flabel(params)

        net.train_split = predefined_split(valid) 
        net.fit(train)
        grid_res = pd.concat([grid_res, history2dframe(net, params)], ignore_index=True)    

        # Get best configuration ..... #
        grid_res.to_pickle(outpath / "grid_results.pkl")


# ----- ----- MAIN ----- ----- #
# ----- ----- ---- ----- ----- #
def split2dataset(split):
    Xlist, y = zip(*split)
    y = torch.stack(y)
    
    X = {}
    for k in Xlist[0].keys():
        v = [x[k] for x in Xlist]
        X[k] = torch.stack(v) if torch.is_tensor(v[0]) else torch.Tensor(v)

    return Dataset(X, y)
    
def trainsplit(dd, ts_perc):
    ddkeys = list(dd.keys())
    n = len(dd[ddkeys[0]])
    idxs = np.random.permutation(range(n))
    tr_size = math.floor(n * (1-ts_perc))
    tr_idxs = idxs[:tr_size]
    ts_idxs = idxs[tr_size:]
    tr, ts = {}, {}

    for k in ddkeys:
        tr[k] = dd[k][tr_idxs]
        ts[k] = dd[k][ts_idxs]
    return tr, ts

def prepare_dataset(df, outpath):
    pr = Cicids2017Preprocessor(flevel=FLEVEL, discretize=False)
    
    validation_day = 3 # Monday
    week_mask = df.index.get_level_values("_time").day != validation_day
    
    df_training = df[week_mask]
    df_training = pr.preprocessing(df_training, fit=True)
    
    df_validation = df[np.bitwise_not(week_mask)]
    df_validation = pr.preprocessing(df_validation)
 
    # validation/test ..... #
    training_windows = defaultdict(list)
    for d in [4, 5, 6, 7]:
        day_df = df_training[df_training.index.get_level_values("_time").day == d]
        day_windows = cb.ts_windowing(day_df, overlapping=WINDOW_OVERLAPPING)

        for k, v in day_windows.items():
            training_windows[k].append(v)

    # Training: all week, Validation: monday
    training = { k: np.concatenate(v) for k, v in training_windows.items() }
    validation = cb.ts_windowing(df_validation, overlapping=WINDOW_OVERLAPPING)

    # Validation attacks: monday + week attacks 
    attacks_training_mask = (training["isanomaly"] == True) & (training["isanomaly"] != "Web Attack")
    attacks_training_mask &= (training["isanomaly"] != "Side_channel")
    attacks_training_mask &= (training["isanomaly"] != "Infiltration")
    attacks_training_mask = np.where(attacks_training_mask)[0]
    validation_attacks = { k: np.concatenate([v, training[k][attacks_training_mask]]) for k, v in validation.items() }
    
    # Training: all week (normal traffic only)
    normal_training_mask = np.where(training["isanomaly"] == False)[0]
    training = { k: x[normal_training_mask] for k, x in training.items() }

    # validation, testing = trainsplit(validation, .33)
    validation_attacks, testing_attacks = trainsplit(validation_attacks, .33) 

    # Validation and testing: monday only (no attacks)
    normal_validation_mask = np.where(validation_attacks["isanomaly"] == False)[0]
    validation = { k: x[normal_validation_mask] for k, x in validation_attacks.items() }

    normal_testing_mask = np.where(testing_attacks["isanomaly"] == False)[0]
    testing = { k: x[normal_testing_mask] for k, x in testing_attacks.items() }

    datasets = { "training": training, "validation": validation, "testing": testing, 
                 "validation_attacks": validation_attacks, "testing_attacks": testing_attacks }
    return datasets


def store_dataset(path, datasets):
    path.mkdir(parents=True, exist_ok=True)
    for k, v in datasets.items():
        cPickle.dump(v, gzip.open(path / f"{k}.pkl", "wb"))


def load_dataset(path):
    datasets = {}
    for f in os.listdir(path):
        if ".pkl" not in f:
            continue
        fname = os.path.splitext(f)[0]
        df = cPickle.load(gzip.open(path / f, "rb"))
        datasets[fname] = df
    return datasets


def Dataset2GPU(dataset):
    if torch.cuda.is_available():
        dataset.X["context"] = dataset.X["context"].cuda()
        dataset.y = dataset.y.cuda()


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description="mEmbedding training")
    parser.add_argument("--datapath", "-d", help="Dataset directory", required=True, type=Path)
    parser.add_argument("--grid", "-g", help="Start grid search", default=False, action="store_true")
    parser.add_argument("--outpath", "-o", help="Grid-search output file", required=True, type=Path)
    args = parser.parse_args()
    args.outpath.mkdir(parents=True, exist_ok=True)

    # Data loading ..... # 
    dataset_files = ["training", "validation", "testing", "validation_attacks", "testing_attacks"]
    timeseries_data = args.datapath / "CICIDS2017_ntop.pkl"
    df = pd.read_pickle(timeseries_data)
    dataset_cache = args.datapath / "cache"
    if dataset_cache.exists():
        datasets = load_dataset(dataset_cache)
    else:
        datasets = prepare_dataset(df, args.outpath)
        store_dataset(dataset_cache, datasets) 

    training = datasets["training"] 
    validation = datasets["validation"]; validation_attacks = datasets["validation_attacks"]

    training = Dataset(*cb.dataset2tensors(training))
    Dataset2GPU(training)
    validation = Dataset(*cb.dataset2tensors(validation))
    Dataset2GPU(validation)
    validation_attacks = Dataset(*cb.dataset2tensors(validation_attacks))
    Dataset2GPU(validation_attacks)

    print(f"training size: {len(training.y)}")
    print(f"validation size: {len(validation.y)}, normal/attacks: {np.unique(validation.y.cpu(), return_counts=True)[1]}")
    print(f"validation_attacks size: {len(validation_attacks.y)}, normal/attacks: {np.unique(validation_attacks.y.cpu(), return_counts=True)[1]}")

    if args.grid: 
        grid_params = ParameterGrid({
            "lr": [ 1e-3, 5e-4, 1e-4, 5e-5, 1e-5 ],
            "batch_size": [ 1024, 2048, 4096 ],
            "max_epochs": [ 2600 ],
        })
        grid_search(training, validation, grid_params, args.outpath)
    else:
        ts2vec, res = ts2vec_cicids2017(training, validation, validation_attacks, args.outpath)
        torch.save(ts2vec.state_dict(), args.outpath / "ts2vec.torch")
