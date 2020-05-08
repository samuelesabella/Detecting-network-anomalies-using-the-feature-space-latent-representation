from collections import defaultdict
from scipy.stats import truncnorm
from sklearn.manifold import TSNE
from skorch.callbacks import EpochScoring
from tqdm import tqdm
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


COHERENT = torch.tensor([ -1. ], dtype=torch.float32)
INCOHERENT = torch.tensor([ 1. ], dtype=torch.float32) 
NORMAL_TRAFFIC = torch.tensor([ -1. ], dtype=torch.float32)
ATTACK_TRAFFIC = torch.tensor([ 1. ], dtype=torch.float32) 

CONTEXT_LEN = 28
ACTIVITY_LEN = 14


def plot_dict(d, fpath):
    plt.figure()
    for k, v in d.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.savefig(fpath)


# ----- ----- DATA RESHAPING ----- ----- #
# ----- ----- -------------- ----- ----- #
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


def coherent_context_picker(ctx_idx, next_ctx_idx, coherency_bounds, context_windows):
    """
        ctx_idx: current context index 
        next_ctx_idx: index of the successive context with no samples in common with the current one
        coherency_bounds: tuple <lbound, rbound> the boundaries of the coherent range
        context_windows: list of all the context windows
    """
    r = random.random()
    if r < .25: # Sample from the current context
        return (context_windows[ctx_idx], COHERENT)
    if r < .5 and next_ctx_idx < len(context_windows): # Sample from the context of the next activity
        ctx_b = context_windows[next_ctx_idx]
        return (ctx_b, COHERENT)
    if r < .75: # Sample context distant in time
        shift_direction = True if random.random() > 0 else False
        lbound, rbound = coherency_bounds
        if lbound > 0 and (rbound > len(context_windows) or shift_direction):
            # sample context before
            random_shift = random.randint(0, lbound)
        else:
            # sample context after
            random_shift = random.randint(rbound, len(context_windows) - 1)
        x_b = context_windows[random_shift]
        return (x_b, INCOHERENT)
    return (None, INCOHERENT) # Sample from full dataset


def ts_windowing(df, ctx_len=CONTEXT_LEN, actv_len=ACTIVITY_LEN, 
        overlapping=.95, consistency_range=240, pick_coherent_activity=True):
    """
        ctx_len   --  context window length, 14 minutes with 4spm (sample per minutes)
        actv_len  --  activity window length, 7 minutes
        overlapping -- context windowing overlapping
        consistency_range  --  activity within this range are considered consistent
    """
    samples = defaultdict(list)
    window_stepsize = max(int(ctx_len * (1 - overlapping)), 1) 

    logging.debug("Windowing time series for each host")
    host_ts = df.groupby(level=['device_category', 'host'])
    for (_, host), ts in tqdm(host_ts):
        # Building context/activity windows ..... #
        ctx_wnds = mit.windowed(range(len(ts)), ctx_len, step=window_stepsize)
        ctx_wnds = filter(lambda x: None not in x, ctx_wnds)
        ctx_wnds_values = list(map(lambda x: ts.iloc[list(x)], ctx_wnds))
        actv_wnds = map(lambda x: random_sublist(x, actv_len), ctx_wnds_values)

        # Coherency and training tuple generation ...... #
        def coh_aus(i):
            next_ctx = i + int(ctx_len / window_stepsize)
            next_incoherent = i + math.ceil(consistency_range / window_stepsize)
            prev_incoherent = i - consistency_range
            coherence_bounds = (prev_incoherent, next_incoherent)
            return coherent_context_picker(i, next_ctx, coherence_bounds, ctx_wnds_values) 
        coherent_contexts = map(coh_aus, range(len(ctx_wnds_values)))
        h_samples = zip(ctx_wnds_values, actv_wnds, coherent_contexts)
        for ctx, activity, (coh_ctx, coh_label) in h_samples:
            samples["activity"].append(activity)
            samples["context"].append(ctx)
            # Need host information to pick samples from different host
            coh_info = host if coh_ctx is None else coh_ctx 
            samples["coherency_context"].append(coh_info)

            samples["coherency"].append(coh_label)
            if "attack" in ctx:
                ctx_attack = NORMAL_TRAFFIC if (ctx["attack"]=="none").all() else ATTACK_TRAFFIC
                samples["attack"].append(ctx_attack)
            
    # Picking random coherency contexts ..... #
    def coh_ctx_to_activity(x):
        # Sampling series from different host
        while isinstance(x, str):
            rctx = random.choice(samples["context"])
            coh_host = rctx.index.get_level_values("host")[0]
            if coh_host != x:
                x = rctx
        if pick_coherent_activity:
            return random_sublist(x, actv_len)
        return x
    logging.debug("Generating coherent activities")
    samples["coherency_activity"] = list(map(coh_ctx_to_activity, tqdm(samples["coherency_context"])))
    del samples["coherency_context"]
    
    # Merging data frames ..... #
    logging.debug("Merging dataframes")
    samples_len = len(samples["context"])
    context_samples = pd.concat(samples["context"], keys=range(samples_len), names=["sample_idx"])
    del samples["context"]
    activity_samples = pd.concat(samples["activity"], keys=range(samples_len), names=["sample_idx"])
    del samples["activity"]
    coherency_activity_samples = pd.concat(samples["coherency_activity"], keys=range(samples_len), names=["sample_idx"])
    del samples["coherency_activity"]

    samples["X"] = pd.concat(
        [activity_samples, context_samples, coherency_activity_samples], 
        keys=["activity", "context", "coherency_activity"], names=["model_input"])
    samples["X"].reset_index(level=["host", "_time", "device_category"], inplace=True)
    samples["X"] = samples["X"].swaplevel(0, 1)

    # Concatenating labels ..... #
    if "attack" in samples:
        samples["attack"] = torch.stack(samples["attack"])
    samples["coherency"] = torch.stack(samples["coherency"])

    return samples


def X2split_tensors(X):
    return { x[0]: X2tensor(x[1]) for x in X.groupby(level=1) }


def X2tensor(X):
    clean_values = X.drop(columns=["_time", "host", "device_category", "attack"])
    ts_values = clean_values.groupby(level="sample_idx").apply(lambda x: x.values)
    return torch.tensor(ts_values).float()

def gpu_if_available(X, Y={}):
    if torch.cuda.is_available():
        X_gpu = { k: v.cuda() for k, v in X.items() }
        Y_gpu = { k: v.cuda() for k, v in Y.items() }
        return X_gpu, Y_gpu
    return X, Y


# ----- ----- LOSSES/SCORING ----- ----- #
# ----- ----- -------------- ----- ----- #
class Contextual_Coherency():
    def __init__(self, alpha=.3):
        self.alpha = alpha
        self.beta_1 = .2
        self.beta_2 = .4

    def __call__(self,  model_output, labels):
        e_actv, e_ctx, e_coh = model_output
        coh_label = labels["coherency"]
        
        ctx_dist = torch.norm((e_actv - e_ctx), p=2, dim=1)        
        contextual_t = F.relu(ctx_dist - self.beta_1)

        mid_point = (e_actv + e_coh) / 2
        mid_point_norm = torch.norm(mid_point, p=2, dim=1)
        label_2_margin = - (coh_label - 1) / 2 # 0 if incoherent, 1 coherent 
        coherency_t = (mid_point_norm * coh_label) + (self.beta_2 * label_2_margin) # Minimize if incoherent, maximize towards beta_2 if coherent
        coherency_t = F.relu(coherency_t)
        
        ctx_coh = (self.alpha * contextual_t) + ((1 - self.alpha) * coherency_t)
        
        return torch.mean(ctx_coh) 


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

    def plot_dist(self, net, X, y, label):
        with torch.no_grad():
            e_actv, e_ctx, e_cohactv = net.forward(X)
        y = y.cpu()
        emb_dist = torch.norm(e_actv - e_cohactv, p=2, dim=1).cpu()
        # Coherent activity
        coherent_idx = torch.where(y==COHERENT)[0]
        coh_mean_dist = torch.mean(emb_dist[coherent_idx])
        self.history["coherent_dist"].append(coh_mean_dist)
        # Incoherent activity
        incoh_idx = torch.where(y==INCOHERENT)[0]
        incoh_mean_dist = torch.mean(emb_dist[incoh_idx])
        self.history["incoherent_dist"].append(incoh_mean_dist)

        plot_dict(self.history, f"{self.path.absolute()}/{label}_distances.png")

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None):
        self.plot_dist(net, dataset_train.X, dataset_train.y["coherency"], "train")
        self.plot_dist(net, dataset_valid.X, dataset_valid.y["coherency"], "valid")


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
            return fsarg[self.label].cpu()
        # Scorer callback
        X, _ = gpu_if_available(dset.X)
        with torch.no_grad():
            if self.label == "attack":
                y_hat = fsarg.module_.context_anomaly(X["context"])
            else:
                y_hat = fsarg.module_.activity_coherency(X["activity"], X["coherency_activity"])
            if y_hat.is_cuda:
                y_hat = y_hat.cpu()

        y_cat = np.maximum(y, 0)
        y_hat_cat = np.round(y_hat)
        res = self.measure(y_cat, y_hat_cat)
        return res


# ----- ----- MODEL ----- ----- #
# ----- ----- ----- ----- ----- #
class Ts2Vec(torch.nn.Module):
    def context_anomaly(self, contexts):
        raise NotImplementedError()

    def toembedding(self, x):
        raise NotImplementedError()

    def forward(self, activity=None, context=None, coherency_activity=None):
        raise NotImplementedError()

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
        self.rnn = nn.GRU(input_size=36, hidden_size=80, num_layers=1, batch_first=True)#256, num_layers=3, batch_first=True)
        self.embedder = nn.Sequential(
            nn.Linear(80, 128),# 256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU())
   
    def context_anomaly(self, ctx):
        a1 = ctx[:, :ACTIVITY_LEN]
        a2 = ctx[:, ACTIVITY_LEN:]
        return self.activity_coherency(a1, a2)

    def activity_coherency(self, a1, a2):
        # 1. => incoherent, 0. => coherent
        e_a1 = self.toembedding(a1)
        e_a2 = self.toembedding(a2)

        mid_point = (e_a1 + e_a2) / 2
        mid_point_norm = torch.norm(mid_point, p=2, dim=1)
        incoherence = F.relu(1. - mid_point_norm)
        return incoherence

    def toembedding(self, x):
        rnn_out, _ = self.rnn(x)
        e = self.embedder(rnn_out[:, -1])
        e = F.normalize(e, p=2, dim=1)
        return e

    def forward(self, activity=None, context=None, coherency_activity=None):
        e_actv = self.toembedding(activity)
        e_ctx = self.toembedding(context)
        e_cohactv = self.toembedding(coherency_activity)
        return (e_actv, e_ctx, e_cohactv)

