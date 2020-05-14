from collections import defaultdict
import copy
from scipy.stats import truncnorm
from sklearn.manifold import TSNE
from skorch.callbacks import EpochScoring
from sklearn import preprocessing
from tqdm import tqdm
import sys
import math
import matplotlib.pyplot as plt
import more_itertools as mit
import numpy as np
import pandas as pd
import random
import skorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
torch.set_default_dtype(torch.float64)


COHERENT = np.array([ -1. ])
INCOHERENT = np.array([ 1. ]) 
NORMAL_TRAFFIC = np.array([ -1. ])
ATTACK_TRAFFIC = np.array([ 1. ]) 

CONTEXT_LEN = 120
ACTIVITY_LEN = 28

BETA_1 = .2
BETA_2 = .4


def plot_dict(d, fpath):
    plt.figure()
    for k, v in d.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.savefig(fpath)


# ----- ----- DATA RESHAPING ----- ----- #
# ----- ----- -------------- ----- ----- #
def ts_windowing(df, overlapping=.95):
    """
        ctx_len   --  context window length, 14 minutes with 4spm (sample per minutes)
        actv_len  --  activity window length, 7 minutes
        overlapping -- context windowing overlapping
        consistency_range  --  activity within this range are considered consistent
    """
    samples = defaultdict(list)
    window_stepsize = max(int(CONTEXT_LEN * (1 - overlapping)), 1) 
    actvstart = int((CONTEXT_LEN - ACTIVITY_LEN) / 2)
    actvend = actvstart + ACTIVITY_LEN

    logging.debug("Windowing time series for each host")
    host_ts = df.groupby(level=['device_category', 'host'])
    for (_, host), ts in tqdm(host_ts):
        # Building context/activity windows ..... #
        wnds = mit.windowed(range(len(ts)), CONTEXT_LEN, step=window_stepsize)
        wnds = filter(lambda x: None not in x, wnds)
        wnds_values = map(lambda x: ts.iloc[list(x)].reset_index(), wnds)
        
        for host_actv in wnds_values:
            context = host_actv.drop(columns=["_time", "host", "device_category", "attack"]).values
            activity = context[actvstart:actvend]
            samples["activity"].append(activity)
            samples["context"].append(context)
            samples["host"].append(host)
            samples["start_time"].append(host_actv["_time"].min().timestamp())
            samples["end_time"].append(host_actv["_time"].max().timestamp())

            activity_attack = NORMAL_TRAFFIC if (host_actv["attack"]=="none").all() else ATTACK_TRAFFIC
            samples["attack"].append(activity_attack)
    
    samples = { k: np.stack(v) for k, v in samples.items() }
    return samples


def dataset2tensors(dataset):
    dataset["activity"] = torch.Tensor(dataset["activity"])
    # Host to id
    dataset["host"] = preprocessing.LabelEncoder().fit_transform(dataset["host"])
    Y = torch.Tensor(dataset["attack"])
    del dataset["attack"]

    if torch.cuda.is_available():
        dataset["activity"] = dataset["activity"].cuda()
        Y = Y.cuda()
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


def filter_distances(current_idx, distances, start_time, end_time, host):
    """Select the minimum above a threshold {th1} and in time range
    """
    time_mask = []
    distances = distances.detach().numpy()

    delay_before = (end_time - start_time[current_idx]) // 3600
    delay_after = (end_time[current_idx] - start_time) // 3600
    delays = torch.stack([delay_before, delay_after]).max(dim=0)[0]
    delay_mask = (delays >= 1).numpy() # cpu().numpy()
    host_mask = (host != host[current_idx])

    valid_idx = np.where(host_mask & (distances > BETA_1))[0]
    return valid_idx[distances[valid_idx].argmin()]


def tuple_mining(e_actv, context, start_time, end_time, host):
    # Anchor positives ..... #
    r = int((CONTEXT_LEN - ACTIVITY_LEN) * zero_one_normal())
    ap = context[:, r:r+ACTIVITY_LEN] 

    # Anchor negatives ..... #
    # Computing distance matrix
    n = len(e_actv)
    dm = torch.pdist(e_actv)
    # Converting tu full nxn matrix
    tri = torch.zeros((n, n))
    tri[np.triu_indices(n, 1)] = dm # .cpu()
    fmatrix = torch.tril(tri.T, 1) + tri
    # Removing diagonal
    fmatrix += sys.maxsize * (torch.eye(n, n))
    # Getting the minimum
    idxs = [filter_distances(i, row, start_time, end_time, host) for i, row in enumerate(fmatrix)]
    dn = torch.stack([e_actv[i] for i in idxs])
    
    if torch.cuda.is_available():
        return ap.cuda(), dn.cuda()
    return ap, dn


# ----- ----- SCORING ----- ----- #
# ----- ----- ------- ----- ----- #
class EpochPlot(skorch.callbacks.Callback):
    def __init__(self, path, onlabel):
        self.path = path
        self.label = onlabel

    @property
    def __name__(self):
        return f"{self.path}/{self.onlabel}.jpg"

    def on_epoch_end(self, net, *args, **kwargs):
        to_plot = { l: [h[l] for h in net.history_] for l in self.label }
        plot_dict(to_plot, f"{self.path.absolute()}/{'_'.join(self.label)}.png")


class DistPlot(skorch.callbacks.Callback):
    def __init__(self, path):
        self.path = path

    @property
    def __name__(self):
        return f"distance_visualization"

    def on_train_begin(self, *args, **kwargs):
        self.history = defaultdict(list)

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
        plot_dict(to_plot, f"{self.path.absolute()}/{label}_distances.png")

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None):
        self.plot_dist(net, dataset_train.X, "train")
        self.plot_dist(net, dataset_valid.X, "valid")


class Ts2VecScore():
    def __init__(self, measure, onlabel):
        self.measure = measure
        self.label = onlabel

    @property
    def __name__(self):
        return f"{self.label}__{self.measure.__name__}"

    def epoch_score(self):
        es_tr = EpochScoring(self, target_extractor=self, 
                             on_train=True, lower_is_better=False, 
                             name=f"train_{self.__name__}")
        es_vl = EpochScoring(self, target_extractor=self, 
                             on_train=False, lower_is_better=False, 
                             name=f"valid_{self.__name__}")
        return es_tr, es_vl

    def __call__(self, fsarg, dset=None, y=None):
        # Extractor called
        if isinstance(fsarg, dict):
            return fsarg[self.label]
        # Scorer callback
        X, _ = dset.X
        with torch.no_grad():
            y_hat = fsarg.module_.context_anomaly(X["context"])
            if y_hat.is_cuda:
                y_hat = y_hat.cpu()

        y_cat = np.maximum(y, 0)
        y_hat_cat = np.round(y_hat)
        res = self.measure(y_cat, y_hat_cat)
        return res


# ----- ----- MODEL ----- ----- #
# ----- ----- ----- ----- ----- #
class Ts2Vec(torch.nn.Module):
    def toembedding(self, x):
        raise NotImplementedError()

    def forward(self, activity=None, context=None, coherency_activity=None):
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

    def to2Dmap(self, df, wlen=ACTIVITY_LEN):
        res_map = pd.DataFrame()
        for (dev_cat, host), ts in df.groupby(level=["device_category", "host"]):
            # Windowing and tensorizing ..... #
            activity_wnds = mit.windowed(range(len(ts)), wlen, step=wlen)
            activity_wnds = filter(lambda x: None not in x, activity_wnds)
            activity_wnds_values = map(lambda x: ts.iloc[list(x)], activity_wnds)
            activity_wnds_values = list(activity_wnds_values)
            host_samples = pd.concat(activity_wnds_values, 
                                     keys=range(len(activity_wnds_values)), names=["sample_idx"])
            host_samples.reset_index(level=["host", "_time", "device_category"], inplace=True)
            sample_tensors = X2tensor(host_samples).detach()
            ebs = self.toembedding(sample_tensors)
            # t-SNE reduction ..... #
            ebs2D = TSNE(n_components=2).fit_transform(ebs.detach())
            ebs2Ddf = pd.DataFrame(ebs2D, columns=[f"x{i}" for i in range(ebs2D.shape[1])])

            # Zipping times with embeddings ..... #
            def min_max_series(x):
                return pd.Series([x["_time"].min(), x["_time"].max()], index=["start", "stop"])

            def mean_timestamp(v):
                t1, _  = v
                return (t1 + pd.Series(v).diff().divide(2)).iloc[1]

            sample_groups = host_samples.groupby(level="sample_idx").apply(min_max_series)
            sample_groups = pd.concat([sample_groups, ebs2Ddf], axis=1, sort=False)
            sample_groups_time_idx = sample_groups.apply(lambda x: mean_timestamp(x[["start", "stop"]].values), axis=1)
            mlt_idx = pd.MultiIndex.from_tuples([(dev_cat, host, t) for t in sample_groups_time_idx])
            sample_groups = sample_groups.set_index(mlt_idx) 
            sample_groups = sample_groups.rename_axis(index=["device_category", "host", "time"])
            res_map = pd.concat([res_map, sample_groups])
        return res_map


class Ts2LSTM2Vec(Ts2Vec):
    def __init__(self):
        super(Ts2LSTM2Vec, self).__init__() 
        self.cnn = nn.Sequential(
            nn.Conv1d(36, 64, 5, 2),
            nn.Conv1d(64, 64, 5, 2),
            nn.Conv1d(64, 128, 4, 1)
        )
        self.embedder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU())
   

    def toembedding(self, x):
        x = self.cnn(x.permute(0, 2, 1)).squeeze()
        e = self.embedder(x)
        e = F.normalize(e, p=2, dim=1)
        return e

    def forward(self, activity=None, context=None, start_time=None, end_time=None, host=None):
        e_actv = self.toembedding(activity)
        with torch.no_grad():
            ap, e_an = tuple_mining(e_actv, context, start_time, end_time, host)
        e_ap = self.toembedding(ap)
        return (e_actv, e_ap, e_an) 

