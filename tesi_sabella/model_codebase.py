from collections import defaultdict
from scipy.stats import truncnorm
from sklearn.manifold import TSNE
from skorch.callbacks import EpochScoring
import logging
from tqdm import tqdm
import math
import more_itertools as mit
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


COHERENT = torch.tensor([1., 0.], dtype=torch.float32)
INCOHERENT = torch.tensor([0., 1.], dtype=torch.float32)
NORMAL_TRAFFIC = torch.tensor([1., 0.], dtype=torch.float32)
ATTACK_TRAFFIC = torch.tensor([0., 1.], dtype=torch.float32) 


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


def coherent_context_picker(current_ctx, ctx_idx, context_windows, step_coherency, min_inconsistency_shift):
    """
    min_inconsistency_shift -- samples to move at least
    """
    r = random.random()
    next_non_overlapping_ctx = ctx_idx + step_coherency + 1
    if r < .25: # Sample from the current context
        return (current_ctx, COHERENT)
    if r < .5 and next_non_overlapping_ctx < len(context_windows): # Sample from the context of the next activity
        ctx_b = context_windows[next_non_overlapping_ctx]
        return (ctx_b, COHERENT)
    if r < .75: # Sample from context distant in time
        shift_direction = True if random.random() > 0 else False
        min_backward_step = ctx_idx - min_inconsistency_shift
        min_forward_step = ctx_idx + len(current_ctx) + min_inconsistency_shift
        if ctx_idx > min_inconsistency_shift and ((ctx_idx > len(context_windows) - min_forward_step) or shift_direction):
            # sample context before
            random_shift = random.randint(0, min_backward_step)
        else:
            # sample context after
            random_shift = random.randint(min_forward_step, len(context_windows) - 1)
        x_b = context_windows[random_shift]
        return (x_b, INCOHERENT)
    return (None, INCOHERENT) # Sample from full dataset


def random_sublist(l, sub_wlen):
    r = int((len(l) - sub_wlen) * zero_one_normal())
    return l.iloc[r:r+sub_wlen]


def ts_windowing(df, w_minutes=14, sub_w_minutes=7, overlapping=.9):
    samples = defaultdict(list)
    # Context window length
    context_wlen = int(w_minutes * 4) # Samples per minutes (one sample every 15 seconds)
    # Activity window length
    activity_wlen = int(sub_w_minutes * 4)
    # 1 hour distance to have inconsistent context (4 samples per minutes)
    stepsize = max(int(context_wlen * (1 - overlapping)), 1) 
    min_inconsistency_dis = math.ceil(4 * 60 / stepsize)

    logging.debug("Windowing time series for each host")
    host_ts = df.groupby(level=['device_category', 'host'])
    for _, ts in tqdm(host_ts):
        # Building context/activity windows ..... #
        ctx_wnds = mit.windowed(range(len(ts)), context_wlen, step=stepsize)
        ctx_wnds = filter(lambda x: None not in x, ctx_wnds)
        ctx_wnds_values = map(lambda x: ts.iloc[list(x)], ctx_wnds)
        ctx_wnds_values = list(ctx_wnds_values)
        actv_wnds = map(lambda x: random_sublist(x, activity_wlen), ctx_wnds_values)

        # Coherency and training tuple generation ...... #
        def coh_aus(ex):
            idx, x = ex
            return coherent_context_picker(x, idx, ctx_wnds_values, activity_wlen, min_inconsistency_dis) 
        coherent_contexts = map(coh_aus, enumerate(ctx_wnds_values))
        h_samples = zip(ctx_wnds_values, actv_wnds, coherent_contexts)
        for ctx, activity, (coh_ctx, coh_label) in h_samples:
            samples["activity"].append(activity)
            samples["context"].append(ctx)
            samples["coherency_context"].append(coh_ctx)

            samples["coherency"].append(coh_label)
            if "attack" in ctx:
                ctx_attack = NORMAL_TRAFFIC if (ctx["attack"]=="none").all() else ATTACK_TRAFFIC
                samples["attack"].append(ctx_attack)
            
    # Picking random coherency contexts ..... #
    def coh_ctx_to_activity(x):
        if x is None:
            x = random.choice(samples["context"])
        return random_sublist(x, activity_wlen)
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


# ----- ----- LOSSES/SCORING ----- ----- #
# ----- ----- -------------- ----- ----- #
class Contextual_Coherency():
    def __init__(self, alpha=.5):
        self.alpha = alpha

    def __call__(self,  model_output, labels):
        e_actv, e_ctx, coherency_score = model_output
        coh_label = labels["coherency"]
        
        context_loss = torch.norm(e_actv - e_ctx, 2)
        coherency_loss = F.binary_cross_entropy(coherency_score, coh_label)
        ctx_coh = (self.alpha * context_loss) + ((1 - self.alpha) * coherency_loss)
        return ctx_coh 


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

    def __call__(self, fsarg, X=None, y=None):
        # Extractor called
        if isinstance(fsarg, dict):
            return fsarg[self.label]
        # Scorer callback
        with torch.no_grad():
            if self.label == "attack":
                y_hat = fsarg.module_.context_anomaly(X.X["context"])
            else:
                _, _, y_hat = fsarg.forward(X)

        y_cat = np.argmax(y, axis=1)
        y_hat_cat = np.argmax(np.round(y_hat), axis=1)
        res = self.measure(y_cat, y_hat_cat)
        return res


# ----- ----- MODEL ----- ----- #
# ----- ----- ----- ----- ----- #
class Ts2Vec(torch.nn.Module):
    def __init__(self):
        super(Ts2Vec, self).__init__()
        self.embedder = nn.LSTM(37, 128, 3)
        self.coherency = nn.Sequential(
            nn.Linear(128, 2),
            nn.Softmax())
    
    def context_anomaly(self, contexts):
        e_a1 = self.toembedding(contexts[:, :28])
        e_a2 = self.toembedding(contexts[:, 28:])
        return self.anomaly_score(e_a1, e_a2)

    def anomaly_score(self, e_a1, e_a2):
        return self.coherency((e_a1 + e_a2) / 2)

    def toembedding(self, x):
        return self.embedder(x)[0][:, -1]

    def to2Dmap(self, df, wlen_minutes=7):
        wlen = wlen_minutes * 4
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

    def forward(self, activity=None, context=None, coherency_activity=None):
        e_actv = self.toembedding(activity)
        e_ctx = self.toembedding(context.detach())
        e_cohactv = self.toembedding(coherency_activity.detach())
        coh_score = self.anomaly_score(e_actv, e_cohactv)
        
        return (e_actv, e_ctx, coh_score)
