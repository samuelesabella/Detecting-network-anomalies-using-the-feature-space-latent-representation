from collections import defaultdict
from pathlib import Path
import sklearn.metrics as skmetrics
from sklearn.model_selection import GridSearchCV, KFold
from skorch.callbacks import EarlyStopping
from skorch.dataset import CVSplit
from skorch.helper import predefined_split
from skorch.helper import SliceDataset
from skorch.net import NeuralNet
import AnchoredTs2Vec as tripletloss
import Callbacks
import Seq2Seq as autoencoder
import AnomalyDetector as ad
import _pickle as cPickle
import argparse
import data_generator as generator
import gzip
import numpy as np
import os
import pandas as pd
import random
import torch


# Reproducibility .... #
SEED = 11769
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# CONSTANTS ..... #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.DoubleTensor)

PATIENCE = 25
MAX_EPOCHS = 250

WINDOW_OVERLAPPING = .45 # .95
FLEVEL = "MAGIK"
DISCRETIZED = False
CONTEXT_LEN = 80 # context window length, 20 minutes with 4spm (sample per minutes) 


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
        
        df["_isanomaly"] = "none"
        host_idxs = df.index.get_level_values("_host")
        times_idxs = df.index.get_level_values("_time")
        
        for atype, adetail, start, stop, host in ATTACKS:
            dt_start = to_dt(start)
            dt_stop = to_dt(stop)
            if host == "*":
                host_selector = True
            else:
                host_selector = (host_idxs==host)
            time_selector = (times_idxs > dt_start) & (times_idxs < dt_stop)
            df.loc[host_selector & time_selector, "_isanomaly"] = f"{atype}__{adetail}"
        return df
    
    def preprocessing(self, df, fit=False):
        # Filtering hosts ..... #
        df = df[df.index.get_level_values("host").str.contains("192.168.10.")]
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
            daily_df = df[df.index.get_level_values("_time").day == d]
            daily_df_preproc = super().preprocessing(daily_df)
            days_df.append(daily_df_preproc)
        preproc_df = pd.concat(days_df)

        if self.compute_discrtz: 
            preproc_df = self.discretize(preproc_df, fit)
         
        preproc_df.index.rename(["_device_category", "_host", "_time"], inplace=True)
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
        return new_samples["_host"].map(self.device_map)


# ----- ----- DATASET ----- ----- #
# ----- ----- ------- ----- ----- #
def cicids2017_partitions(df):
    pr = Cicids2017Preprocessor(flevel="MAGIK", discretize=False)
    
    monday = 3
    week_days = [4, 5, 6, 7]
    week_mask = df.index.get_level_values("_time").day != monday
    tserver_mask = df.index.get_level_values("host") != "192.168.10.50"
    
    TR_mask = week_mask & tserver_mask
    DT_mask = np.bitwise_not(tserver_mask)
    M_mask = np.bitwise_not(week_mask) & tserver_mask

    df_TR = pr.preprocessing(df[TR_mask])
    TS = pr.preprocessing(df[M_mask])
    DT = pr.preprocessing(df[DT_mask])
    
    # Split TR by day
    TR = []
    for d in week_days:
        day_df = df_TR[df_TR.index.get_level_values("_time").day == d]
        TR.append(day_df)
    
    return { "TR": TR, "TS": TS, "DT": DT }


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


# ----- ----- EXPERIMENTS ----- ----- #
# ----- ----- ----------- ----- ----- #
def history2dframe(net, labels=None):
    ignore_keys = [ "batches", "train_batch_count", "valid_batch_count" ]
    best_epoch = next(x for x in reversed(net.history) if x["valid_loss_best"])
    if labels:
        best_epoch.update({ f"hyperparam_{k}": v for k, v in labels.items() })

    s = pd.Series(best_epoch).to_frame().T
    s = s.drop(columns=ignore_keys)
    s = s.drop([c for c in s.columns if "best" in c], axis=1)
    
    s["ctx_overlapping"] = WINDOW_OVERLAPPING
    s["features"] = FLEVEL
    s["ctx_len"] = CONTEXT_LEN
    s["discretized"] = DISCRETIZED

    return s.infer_objects()


def configureAnchor(outpath, checkpoint: Path = None):
    batch_size = 4096
    lr = 1e-4
    model_args = {
        "module__pool": "last",
        "module__input_size": 19,
        "module__rnn_size": 128,
        "module__rnn_layers": 3,
        "module__latent_size": 128
    }

    dist_plot = Callbacks.DistPlot(outpath)
    loss_plot = Callbacks.EpochPlot(outpath, ["train_loss", "valid_loss"])

    net = ad.WindowedAnomalyDetector(
        tripletloss.GruLinear, tripletloss.ContextualCoherency, 
        optimizer=torch.optim.Adam, 
        iterator_train__shuffle=False,
        lr=lr, batch_size=batch_size, max_epochs=MAX_EPOCHS,
        **model_args,
        device=DEVICE, verbose=1,
        train_split=CVSplit(5, random_state=SEED),
        callbacks=[ dist_plot, loss_plot,
                    EarlyStopping("valid_loss", lower_is_better=True, patience=PATIENCE)])    
    if checkpoint is not None:
        net.initialize_context(CONTEXT_LEN)
        net.initialize()
        state_dict = torch.load(str(checkpoint), map_location=torch.device("cpu"))
        net.module_.load_state_dict(state_dict)

    return net


def configureSeq2Seq(outpath, checkpoint: Path = None):
    batch_size = 64
    lr = 1e-4
    model_args = {
        "module__pool": "mean",
        "module__input_size": 19,
        "module__teacher_forcing_ratio": 7.,
        "module__rnn_layers": 1,
        "module__latent_size": 128
    }

    loss_plot = Callbacks.EpochPlot(outpath, ["train_loss", "valid_loss"])

    net = ad.WindowedAnomalyDetector(
        autoencoder.Seq2Seq, autoencoder.ReconstructionError, 
        optimizer=torch.optim.Adam, 
        iterator_train__shuffle=False,
        lr=lr, batch_size=batch_size, max_epochs=MAX_EPOCHS,
        **model_args,
        device=DEVICE, verbose=1,
        train_split=CVSplit(5, random_state=SEED),
        callbacks=[ loss_plot,
                    EarlyStopping("valid_loss", lower_is_better=True, patience=PATIENCE)])    
    if checkpoint is not None:
        net.initialize_context(CONTEXT_LEN)
        net.initialize()
        state_dict = torch.load(str(checkpoint), map_location=torch.device("cpu"))
        net.module_.load_state_dict(state_dict)

    return net

def ts2vec_cicids2017(net, train, testing, testing_attacks, outpath):
    net.fit(train)
    
    return net.module_, history2dframe(net)


# ----- ----- GRID SEARCH ----- ----- #
# ----- ----- ----------- ----- ----- #
def grid_search(train, grid_params, module, loss, outpath):
    net = NeuralNet(module, loss, 
                    optimizer=torch.optim.Adam, max_epochs=MAX_EPOCHS,
                    device=DEVICE, iterator_train__shuffle=False, 
                    callbacks=[ EarlyStopping("valid_loss", lower_is_better=True, patience=PATIENCE) ],
                    train_split=CVSplit(5, random_state=SEED))#, verbose=0)

    kf = KFold(n_splits=5, shuffle=False, random_state=SEED)
    gs = GridSearchCV(net, grid_params, cv=kf, refit=True, scoring=loss(), verbose=10)

    grid_res = gs.fit(SliceDataset(train), train.y)

    df = pd.DataFrame(grid_res.cv_results_)
    # Adding some meta info
    df["param_feature_set"] = FLEVEL
    df["param_overlapping"] = WINDOW_OVERLAPPING
    df["param_context_len"] = CONTEXT_LEN
    df["param_discretized"] = DISCRETIZED
    df["param_module"] = module.__name__
    fname = f"gridsearch_{FLEVEL}_{CONTEXT_LEN}_{WINDOW_OVERLAPPING}_{DISCRETIZED}_{module.__name__}.pkl"
    df.to_pickle(outpath / fname)


# ----- ----- MAIN ----- ----- #
# ----- ----- ---- ----- ----- #
def describe_datasets(training, testing, detection):
    train_len = len(training.y)
    print(f"TR: {train_len} contexts")
    ts_len = len(testing.y)
    print(f"TS: {ts_len} contexts")
    dt_perc = len(detection[detection["_isanomaly"] != "none"]) / len(detection)
    print(f"DT: {len(detection)} samples, {dt_perc}% attacks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mEmbedding training")
    parser.add_argument("--datapath", "-d", help="Dataset directory", required=True, type=Path)
    parser.add_argument("--technique", "-t", 
                        help="One among triplet loss based model (TL) or auto-encoder (AE)", 
                        required=True, type=str, choices=["AE", "TL"])
    parser.add_argument("--grid", "-g", help="Start grid search", default=False, action="store_true")
    parser.add_argument("--outpath", "-o", help="Grid-search output file", required=True, type=Path)
    parser.add_argument("--reset_data", "-r", help="Recreates the dataset", default=False, action="store_true")
    args = parser.parse_args()
    args.outpath.mkdir(parents=True, exist_ok=True)
    
    # if args.technique == "AE":
    #     CONTEXT_LEN = 15

    # Data loading ..... # 
    dataset_cache = args.datapath / "cache"
    if dataset_cache.exists() and not args.reset_data:
        datasets = load_dataset(dataset_cache)
    else:
        timeseries_data = args.datapath / "CICIDS2017_ntop.pkl"
        df = pd.read_pickle(timeseries_data)
        datasets = cicids2017_partitions(df)
        store_dataset(dataset_cache, datasets) 

    dg = ad.WindowedDataGenerator(WINDOW_OVERLAPPING, CONTEXT_LEN)
    training = dg(datasets["TR"])
    testing = dg([datasets["TS"]])
    detection = datasets["DT"]
    describe_datasets(training, testing, detection)
    input_size = training.X["context"].shape[-1]

    if args.technique=="TL":
        grid_params = {
            "lr": [ 5e-4, 1e-4 ],
            "batch_size": [ 4096 ],
            "module__pool": [ "mean", "last" ],
            "module__input_size": [ input_size ],
            "module__rnn_size": [ 64, 128 ],
            "module__rnn_layers": [ 1, 3 ],
            "module__latent_size": [ 64, 128 ] }
        module = tripletloss.GruLinear
        loss = tripletloss.ContextualCoherency
        net = configureAnchor(args.outpath)
    else:
        grid_params = { 
                "lr": [ 1e-3, 1e-4 ],
                "batch_size": [ 64, 128 ],
                "module__input_size": [ input_size ],
                "module__teacher_forcing_ratio": [ 1., .7 ],
                "module__rnn_layers": [ 1 ],
                "module__latent_size": [ 32, 64, 128 ] }
        module = autoencoder.Seq2Seq
        loss = autoencoder.ReconstructionError 
        net = configureSeq2Seq(args.outpath)
        # Fix target output :-)
        training.y = training.X["context"]
        testing.y = testing.X["context"]

    if args.grid:
        grid_search(training, grid_params, module, loss, args.outpath)
    else:
        ts2vec, _ = ts2vec_cicids2017(net, training, testing, detection, args.outpath)
        print("Terminating, saving the model")
        torch.save(ts2vec.state_dict(), args.outpath / "ts2vec.torch")
