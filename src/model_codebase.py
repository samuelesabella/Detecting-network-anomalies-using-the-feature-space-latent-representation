from collections import defaultdict
from scipy.stats import truncnorm
from skorch.callbacks import EpochScoring
from sklearn import preprocessing
from tqdm import tqdm
import sys
# from umap import UMAP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import more_itertools as mit
import numpy as np
import pandas as pd
import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.float64)


# ----- ----- CONSTANTS ----- ----- #
# ----- ----- --------- ----- ----- #
NORMAL_TRAFFIC = np.array([ 0. ])
ATTACK_TRAFFIC = np.array([ 1. ]) 

CONTEXT_LEN = 80 # context window length, 28 minutes with 4spm (sample per minutes) 
ACTIVITY_LEN = 40 # activity window length, 14 minutes 

# Triplet margins
BETA_1 = .1
BETA_2 = .4


# ----- ----- DATA RESHAPING ----- ----- #
# ----- ----- -------------- ----- ----- #
def dfwindowed(df, wlen, step):
    df_len = len(df)
    wnds = mit.windowed(range(df_len), wlen, step=step)
    wnds = filter(lambda x: None not in x, wnds)
    wnds_values = map(lambda x: df.iloc[list(x)].reset_index(), wnds)
    return wnds_values


def ts_windowing(df, overlapping=.95):
    """
    overlapping -- context windowing overlapping
    consistency_range  --  activity within this range are considered consistent
    """
    samples = defaultdict(list)
    window_stepsize = max(int(CONTEXT_LEN * (1 - overlapping)), 1) 

    logging.debug("Windowing time series for each host")
    host_ts = df.groupby(level=['device_category', 'host'])
    for (device_category, host), ts in tqdm(host_ts):
        # Building context/activity windows ..... #
        windows = dfwindowed(ts, CONTEXT_LEN, window_stepsize)
        for context in windows:
            ctxvalues = context.drop(columns=["_time", "host", "device_category", "isanomaly"]).values
            samples["context"].append(ctxvalues)
            samples["host"].append(host)
            samples["device_category"].append(device_category)
            samples["start_time"].append(context["_time"].min().timestamp())
            samples["end_time"].append(context["_time"].max().timestamp())

            actv1_isnormal = (context[:ACTIVITY_LEN]["isanomaly"] == "none").all()
            actv2_attack_frac = (context[ACTIVITY_LEN:]["isanomaly"] != "none").value_counts()[False] / ACTIVITY_LEN
            actv2_isattack = actv2_attack_frac >= .5
            isanomaly = actv1_isnormal and actv2_isattack 
            # isanomaly = NORMAL_TRAFFIC if (context["isanomaly"]=="none").all() else ATTACK_TRAFFIC
            samples["isanomaly"].append(isanomaly)

            attack_type = "none"
            if isanomaly == ATTACK_TRAFFIC:
                attack_type = np.unique(context["isanomaly"][context["isanomaly"]!="none"])[0]
            samples["attack_type"].append(attack_type)
    samples = { k: np.stack(v) for k, v in samples.items() }
    return samples

def dataset2tensors(dataset):
    dataset["context"] = torch.Tensor(dataset["context"])
    # Host to id
    dataset["host"] = preprocessing.LabelEncoder().fit_transform(dataset["host"])
    dataset["device_category"] = preprocessing.LabelEncoder().fit_transform(dataset["device_category"])
    Y = torch.Tensor(dataset["isanomaly"])
    del dataset["isanomaly"]
    del dataset["attack_type"]

    return dataset, Y


# ----- ----- LOSSES ----- ----- #
# ----- ----- ------ ----- ----- #
class Contextual_Coherency():
    def __call__(self, model_out, labels):
        e_actv, e_ap, e_an = model_out 
        ap_dist = F.relu(torch.norm((e_actv - e_ap), p=2, dim=1) - BETA_1)
        an_dist = F.relu(BETA_2 - torch.norm((e_actv - e_an), p=2, dim=1))
        return torch.mean(ap_dist + an_dist)


# ----- ----- TUPLE MINING ----- ----- #
# ----- ----- ------------ ----- ----- #
class RNTrunc():
    def __init__(self, mean, std, clip_values):
        clip_min, clip_max = clip_values
        p1, p2 = (clip_min - mean) / std, (clip_max - mean) / std
        self.r = truncnorm(p1, p2, loc=mean, scale=std)
    
    def __call__(self, size=None):
        return self.r.rvs(size=size)


zero_one_normal = RNTrunc(.5, .2, (0, 1))


def random_sublist(l, sub_wlen):
    r = int((len(l) - sub_wlen) * zero_one_normal())
    return l.iloc[r:r+sub_wlen]


def fast_filter(distances, discriminator):
    discr_matrix = torch.stack([discriminator]*len(discriminator))
    same_discr = (discr_matrix == discr_matrix.T)
    distances[same_discr] = sys.maxsize

    # semi-hard triplet mining
    # distances[distances < BETA_1] = sys.maxsize-1

    return distances.argmin(axis=1)


def find_neg_anchors(e_actv, e_ap, start_time, end_time, host, device_category):
    """find negative anchors within a batch
    """
    # Computing distance matrix
    n = len(e_actv)
    dm = torch.pdist(e_actv)
    # Converting tu full nxn matrix
    tri = torch.zeros((n, n))
    tri[np.triu_indices(n, 1)] = dm
    fmatrix = torch.tril(tri.T, 1) + tri
    # Removing diagonal
    fmatrix += sys.maxsize * (torch.eye(n, n))
    # Getting the minimum
    idxs = fast_filter(fmatrix, device_category) 
    dn = e_actv[idxs]
    
    return dn


# ----- ----- TRAINING CALLBACKS ----- ----- #
# ----- ----- ------------------ ----- ----- #
def plot_dict(d, fpath):
    plt.clf()
    for k, v in d.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.savefig(fpath)


class EpochPlot(skorch.callbacks.Callback):
    def __init__(self, path, on_measure):
        self.path = path
        self.flabel = ""
        self.on_measure = on_measure

    @property
    def __name__(self):
        return '_'.join(self.on_measure)

    def set_flabel(self, d):
        self.flabel = "__".join([f"{k}_{v}" for k, v in d.items()])
    
    def on_epoch_end(self, net, *args, **kwargs):
        to_plot = { l: [h[l] for h in net.history_] for l in self.on_measure }
        fname =  f"{self.path.absolute()}/{self.flabel}__{self.__name__}.png"
        plot_dict(to_plot, fname)


class DistPlot(skorch.callbacks.Callback):
    def __init__(self, path):
        self.path = path
        self.flabel = "distance"

    @property
    def __name__(self):
        return f"distance_visualization"

    def on_train_begin(self, *args, **kwargs):
        self.history = defaultdict(list)
    
    def set_flabel(self, d):
        self.flabel = "__".join([f"{k}_{v}" for k, v in d.items()])
    
    def plot_dist(self, net, X, label):
        with torch.no_grad():
            e_actv, e_ap, e_an = net.forward(X)
        # Coherent activity
        coh_dist = torch.norm((e_actv - e_ap), p=2, dim=1).cpu()
        coh_mean_dist = torch.mean(coh_dist)
        self.history[f"coherent_dist_{label}"].append(coh_mean_dist)
        # Incoherent activity
        incoh_dist = torch.norm((e_actv - e_an), p=2, dim=1).cpu()
        incoh_mean_dist = torch.mean(incoh_dist)
        self.history[f"incoherent_dist_{label}"].append(incoh_mean_dist)

        keys = [f"coherent_dist_{label}", f"incoherent_dist_{label}"]
        to_plot = { k: v for k, v in self.history.items() if k in keys }
        fname = f"{self.path.absolute()}/{self.flabel}__{label}_distances.png"
        plot_dict(to_plot, fname)

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None):
        self.plot_dist(net, dataset_train.X, "train")
        self.plot_dist(net, dataset_valid.X, "valid")


class Ts2VecScore():
    def __init__(self, measure):
        self.measure = measure

    @property
    def __name__(self):
        return f"{self.measure.__name__}"

    def epoch_score(self, on_train=None):
        if on_train is not None:
            tv = "train" if on_train else "valid"
            score_name = f"{tv}_{self.__name__}"
            es_vl = EpochScoring(self, on_train=on_train, lower_is_better=False, 
                                 name=score_name)
            return es_vl

        tr_score = EpochScoring(self, on_train=True, lower_is_better=False, 
                                 name=f"train_{self.__name__}")
        vl_score = EpochScoring(self, on_train=False, lower_is_better=False, 
                                 name=f"valid_{self.__name__}")
        return (tr_score, vl_score)

    def __call__(self, net, dset=None, y=None):
        with torch.no_grad():
            y_hat = net.module_.context_anomaly(dset.X["context"])
        res = self.measure(y, np.round(y_hat.cpu()))
        return res


# ----- ----- MODELS ----- ----- #
# ----- ----- ------ ----- ----- #
class AnchorTs2Vec(torch.nn.Module):
    def toembedding(self, x):
        raise NotImplementedError()

    def context_anomaly(self, ctx):
        a1 = ctx[:, :ACTIVITY_LEN]
        a2 = ctx[:, ACTIVITY_LEN:]
        return self.activity_coherency(a1, a2)

    def activity_coherency(self, a1, a2):
        # 1. => incoherent, 0. => coherent
        e_a1 = self.toembedding(a1)
        e_a2 = self.toembedding(a2)

        dist = (torch.norm(e_a1 - e_a2, p=2, dim=1) - BETA_1) / BETA_2
        return torch.clamp(dist, 0., 1.)

    def forward(self, context=None, device_category=None, start_time=None, end_time=None, host=None):
        actv = context[:, :ACTIVITY_LEN]
        e_actv = self.toembedding(actv) 

        with torch.no_grad():
            ap = context[:, ACTIVITY_LEN:] 
            e_ap = self.toembedding(ap)
            e_an = find_neg_anchors(e_actv, e_ap, start_time, end_time, host, device_category)
        return (e_actv, e_ap, e_an)


class STC(AnchorTs2Vec):
    def __init__(self):
        super(STC, self).__init__() 
        self.rnn = nn.GRU(input_size=11, hidden_size=32, num_layers=1, batch_first=True)
        self.embedder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU())

    def toembedding(self, x):
        rnn_out, _ = self.rnn(x)
        e = self.embedder(rnn_out[:, -1])
        # e = F.normalize(e, p=2, dim=1)
        return e

