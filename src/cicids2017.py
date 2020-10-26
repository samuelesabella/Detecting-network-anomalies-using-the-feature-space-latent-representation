from collections import defaultdict
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
PATIENCE = 250
FLEVEL = "NF_BLMISC"

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
        df = df.drop(["host_unreachable_flows:flows_as_client",
                      "dns_qry_sent_rsp_rcvd:replies_error_packets",
                      "dns_qry_rcvd_rsp_sent:replies_error_packets"], axis=1) # All zeros in the dataset
        # Removing initial non zero traffic ..... #
        index_hours = df.index.get_level_values("_time").hour
        working_hours = (index_hours > 8) & (index_hours < 17)
        df = df[working_hours]
        # Passing the ball ..... #
        preproc_df = super().preprocessing(df, **kwargs)
    
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


def ts2vec_cicids2017(train, test, outpath):
    dist_plot = cb.DistPlot(outpath)
    loss_plot = cb.EpochPlot(outpath, ["train_loss", "valid_loss"])
    net = NeuralNet(
        cb.STC, LOSS, optimizer=torch.optim.Adam, 
        iterator_train__shuffle=True,
        lr=5e-4, batch_size=512, max_epochs=1,
        device=DEVICE, verbose=1, train_split=None,
        callbacks=[
            dist_plot, loss_plot,
            cb.Ts2VecScore(skmetrics.accuracy_score).epoch_score(),
            cb.Ts2VecScore(skmetrics.recall_score).epoch_score(),
            cb.Ts2VecScore(skmetrics.precision_score).epoch_score(),
            cb.Ts2VecScore(skmetrics.roc_auc_score).epoch_score(),
            EarlyStopping("valid_loss", lower_is_better=True, patience=PATIENCE)
        ])    

    # Retrain on whole dataset ..... #
    net.train_split = predefined_split(test)
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
    

def prepare_dataset(df, outpath):
    pr = Cicids2017Preprocessor(flevel=FLEVEL)
    
    # unsupervised data ..... #
    df_monday = df[df.index.get_level_values("_time").day == 3]
    df_monday = pr.preprocessing(df_monday, update=True)
    monday = cb.ts_windowing(df_monday, overlapping=WINDOW_OVERLAPPING)
    monday = Dataset(*cb.dataset2tensors(monday))

    # Storing features metainfo ..... #
    with open(outpath / "features.txt", "w+") as f:
        f.write(f"{FLEVEL}\n")
        f.write(f"---------- \n")
        f.write("\n".join(df_monday.columns))
    
    # validation/test ..... #
    labeled_samples = defaultdict(list)
    for d in [4, 5, 6, 7]:
        df_day = df[df.index.get_level_values("_time").day == d]
        df_day_preproc = pr.preprocessing(df_day, update=False)
        day_samples = cb.ts_windowing(df_day_preproc, overlapping=WINDOW_OVERLAPPING)

        for k, v in day_samples.items():
            labeled_samples[k].append(v)
    labeled_samples = { k: np.concatenate(v) for k, v in labeled_samples.items() }
    labeled_samples = Dataset(*cb.dataset2tensors(labeled_samples))

    # Stratified sampling for each attack ..... #
    devlabels = labeled_samples.y.cuda() if torch.cuda.is_available() else labeled_samples.y
    labeled_train, labeled_test = train_test_split(labeled_samples, test_size=0.33, 
                                                   random_state=SEED, stratify=devlabels)
    labeled_train = split2dataset(labeled_train)
    labeled_test = split2dataset(labeled_test)

    return monday, labeled_train, labeled_test 


def store_dataset(monday, labeled_train, ts, path):
    path.mkdir(parents=True, exist_ok=True)
    cPickle.dump(monday, gzip.open(path / "monday.pkl", "wb"))
    cPickle.dump(labeled_train, gzip.open(path / "labeled_mondayain.pkl", "wb"))
    cPickle.dump(labeled_test, gzip.open(path / "labeled_test.pkl", "wb"))


def load_dataset(path):
    monday = cPickle.load(gzip.open(path / "monday.pkl", "rb"))
    labeled_mondayain = cPickle.load(gzip.open(path / "labeled_mondayain.pkl", "rb"))
    labeled_test = cPickle.load(gzip.open(path / "labeled_test.pkl", "rb"))
    return monday, labeled_mondayain, labeled_test


def Dataset2GPU(dataset):
    if torch.cuda.is_available():
        dataset.X["context"] = dataset.X["context"].cuda()
        dataset.y = dataset.y.cuda()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description="mEmbedding training")
    parser.add_argument("--datapath", "-d", help="Dataset directory", required=True, type=Path)
    parser.add_argument("--grid", "-g", help="Start grid search", default=False, action="store_true")
    parser.add_argument("--outpath", "-o", help="Grid-search output file", required=True, type=Path)
    args = parser.parse_args()
    args.outpath.mkdir(parents=True, exist_ok=True)

    # Data loading ..... # 
    timeseries_data = args.datapath / "CICIDS2017_ntop.pkl"
    df = pd.read_pickle(timeseries_data)
    dataset_cache = args.datapath / "cache"
    if dataset_cache.exists():
        monday, labeled_train, labeled_test = load_dataset(dataset_cache)
    else:
        monday, labeled_train, labeled_test = prepare_dataset(df, args.outpath)
        store_dataset(monday, labeled_train, labeled_test, dataset_cache) 

    Dataset2GPU(monday); 
    Dataset2GPU(labeled_train); Dataset2GPU(labeled_test)

    if args.grid: 
        grid_params = ParameterGrid({
            "lr": [ 1e-3, 5e-4, 1e-4, 5e-5, 1e-5 ],
            "batch_size": [ 1024, 2048, 4096 ],
            "max_epochs": [ 2600 ],
        })
        grid_search(monday, labeled_train, grid_params, args.outpath)
    else:
        ts2vec, res = ts2vec_cicids2017(monday, labeled_test, args.outpath)
        torch.save(ts2vec.state_dict(), args.outpath / "ts2vec.torch")
