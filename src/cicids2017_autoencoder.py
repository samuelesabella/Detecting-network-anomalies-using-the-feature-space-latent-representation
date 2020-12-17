from collections import defaultdict
import torch.nn as nn
from pathlib import Path
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from cicids2017 import prepare_dataset, store_dataset, load_dataset, Dataset2GPU
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from skorch.callbacks import EarlyStopping
from skorch.dataset import CVSplit
import skorch
from skorch.dataset import Dataset
from skorch.helper import SliceDataset
from skorch.helper import predefined_split
from skorch.net import NeuralNet
from tqdm import tqdm
import _pickle as cPickle
import argparse
import data_generator as generator
import gzip
import itertools
import logging
import math
import model_codebase as cb
import numpy as np
import os
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
PATIENCE = 25
MAX_EPOCHS = 250

WINDOW_OVERLAPPING = .95
FLEVEL = "MAGIK"
DISCRETIZED = False
CONTEXT_LEN = 80 # context window length, 28 minutes with 4spm (sample per minutes) 
ACTIVITY_LEN = 40 # activity window length, 14 minutes 


# ----- ----- PREPROCESSING ----- ----- #
# ----- ----- ------------- ----- ----- #


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


def ts2vec_cicids2017(train, testing, testing_attacks, outpath):
    batch_size = 32
    lr = 5e-4
    model_args = {
        "module__input_size": 19,
        "module__latent_size": 128,
        "module__rnn_layers": 1,
        "module__latent_size": 128
    }

    loss_plot = cb.EpochPlot(outpath, ["train_loss", "valid_loss"])
    rec_prec_plot = cb.EpochPlot(outpath, ["valid_precision_score", "valid_recall_score"])

    net = NeuralNet(
        cb.AE, cb.ReconstructionError, optimizer=torch.optim.Adam, 
        iterator_train__shuffle=False,
        lr=lr, batch_size=batch_size, max_epochs=MAX_EPOCHS,
        **model_args,
        device=DEVICE, verbose=1,
        train_split=CVSplit(5, random_state=SEED),
        callbacks=[
            # cb.Ts2VecScore(skmetrics.recall_score, data=testing_attacks).epoch_score(on_train=False),
            # cb.Ts2VecScore(skmetrics.precision_score, data=testing_attacks).epoch_score(on_train=False),
            # cb.Ts2VecScore(skmetrics.roc_auc_score, data=testing_attacks).epoch_score(on_train=False),
            # loss_plot, rec_prec_plot,
            # EarlyStopping("valid_loss", lower_is_better=True, patience=7
            # EarlyStopping("valid_roc_auc_score", lower_is_better=False, patience=7)
        ])    

    # Retrain on whole dataset ..... #
    Y = torch.stack([x["context"] for x in SliceDataset(train)])
    net.fit(Y, Y)

    # Test Loss ..... #
    loss_ts = cb.Contextual_Coherency()(net.forward(testing.X))
    print(f"Testing loss: {loss_ts}")

    # Detection capabilities ..... #
    X_attacks = testing_attacks.X["context"]
    y_attacks = testing_attacks.y.cpu()
    y_hat = net.module_.context_anomaly(X_attacks).detach()
    y_hat = np.round(y_hat.cpu())
    
    report = classification_report(y_attacks, y_hat)
    metrics_rep = [ metrics.roc_auc_score,
                    metrics.precision_score, metrics.recall_score,
                    metrics.accuracy_score, metrics.f1_score]
    for m in metrics_rep:
        mres = m(y_attacks, y_hat)
        print(f"{m.__name__}(moday+attacks): {mres}")

    tn, fp, fn, tp = metrics.confusion_matrix(y_attacks, y_hat, normalize="all").ravel()
    print("\n Confusion matrix")
    print(f"\ttp: {tp} \tfp: {fp} \n\tfn: {fn} \ttn: {tn}")
    print(f"\n{report}")
    
    return net.module_, history2dframe(net)


# ----- ----- GRID SEARCH ----- ----- #
# ----- ----- ----------- ----- ----- #
def grid_search(train, grid_params, outpath):
    net = NeuralNet(cb.AE, cb.ReconstructionError, optimizer=torch.optim.Adam,
                    max_epochs=MAX_EPOCHS,
                    device=DEVICE, iterator_train__shuffle=False, 
                    callbacks=[ EarlyStopping("valid_loss", lower_is_better=True, patience=PATIENCE) ],
                    train_split=CVSplit(5, random_state=SEED),
                    verbose=0)
    kf = KFold(n_splits=5, shuffle=False, random_state=SEED)
    gs = GridSearchCV(net, grid_params, cv=kf, refit=True, scoring=cb.Contextual_Coherency.as_score, verbose=10)
    grid_res = gs.fit(SliceDataset(train))

    grid_search_df = pd.DataFrame(grid_res.cv_results_)
    # Get best configuration ..... #
    fname = f"AE_gridSearch__features_{FLEVEL}__overlapping_{WINDOW_OVERLAPPING}__ctxlen_{CONTEXT_LEN}__discretized_{DISCRETIZED}.pkl"
    grid_search_df.to_pickle(outpath / fname)


# ----- ----- MAIN ----- ----- #
# ----- ----- ---- ----- ----- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mEmbedding training")
    parser.add_argument("--datapath", "-d", help="Dataset directory", required=True, type=Path)
    parser.add_argument("--grid", "-g", help="Start grid search", default=False, action="store_true")
    parser.add_argument("--outpath", "-o", help="Grid-search output file", required=True, type=Path)
    parser.add_argument("--reset_data", "-r", help="Recreates the dataset", default=False, action="store_true")
    args = parser.parse_args()
    args.outpath.mkdir(parents=True, exist_ok=True)

    # Data loading ..... # 
    dataset_files = ["training", "validation", "testing", "validation_attacks", "testing_attacks"]
    dataset_cache = args.datapath / "cache"
    if dataset_cache.exists() and not args.reset_data:
        datasets = load_dataset(dataset_cache)
    else:
        timeseries_data = args.datapath / "CICIDS2017_ntop.pkl"
        df = pd.read_pickle(timeseries_data)
        datasets = prepare_dataset(df)
        store_dataset(dataset_cache, datasets) 

    training = datasets["training"] 
    testing = datasets["testing"]; 
    testing_attacks = datasets["testing_attacks"]
    input_size = training["context"].shape[-1]

    train_len = len(training["isanomaly"])
    print(f"training (week normal): {train_len} samples")
    val_len = len(testing["isanomaly"])
    print(f"testing (monday only): {val_len} samples")
    val_att_len = len(testing_attacks['isanomaly'])
    val_att_perc = np.unique(testing_attacks["isanomaly"], return_counts=True)[1] / val_att_len
    print(f"testing_attacks (192.168.10.50): {val_att_len} samples, normal/attacks percentage: {val_att_perc}")

    training = Dataset(*cb.dataset2tensors(training))
    Dataset2GPU(training)
    testing = Dataset(*cb.dataset2tensors(testing))
    Dataset2GPU(testing)
    testing_attacks = Dataset(*cb.dataset2tensors(testing_attacks))
    Dataset2GPU(testing_attacks)

    if args.grid: 
        grid_params = {
            "lr": [ 5e-4, 1e-4 ],
            "batch_size": [ 4096, 6000 ],
            "module__input_size": [ input_size ],
            "module__rnn_size": [ 64, 128 ],
            "module__rnn_layers": [ 1, 3 ],
            "module__latent_size": [ 64, 128 ]}
        grid_search(training, grid_params, args.outpath)
    else:
        ts2vec, res = ts2vec_cicids2017(training, testing, testing_attacks, args.outpath)
        print("Terminating, saving the model")
        torch.save(ts2vec.state_dict(), args.outpath / "ts2vec.torch")
