import torch
import torch.nn.functional as F
import more_itertools as mit
from scipy.stats import truncnorm
import random


INC_TENSOR = torch.tensor([0, 1])
COH_TENSOR = torch.tensor([1, 0])


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


def coherency_generator(x, idx, time_windows):
    r = random.random()
    if r < .25: # Sample from the current context
        return (x, COH_TENSOR)
    if r < .5: # Sample from the context of the next activity
        x_b = time_windows[idx + 1]
        return (x_b, COH_TENSOR)
    if r < .75: # Sample from context distant in time
        random_shift = random.randint(1, len(time_windows)-idx)
        if random.random() > .5:
            random_shift = random.randint(-2, -idx) 
        x_b = time_windows[idx + 1 + random_shift]
        return (x_b, INC_TENSOR)
    return (None, INC_TENSOR) # Sample from full dataset


def random_sublist(l, sub_wlen):
    r = random.randint(0, len(l)-sub_wlen)
    return l[r:r+sub_wlen]
    

def context_merge(ctx_a, ctx_b):
    # TODO: Multivariate merge
    r = zero_one_normal()
    r_idx = int(len(ctx_a) * r)
    return (ctx_a[:r_idx] + ctx_b[r_idx:])


def ts_windowing(df, w_minutes=15, sub_w_minutes=7):
    X = []
    wlen = int(w_minutes * 4) # Samples per minutes (one sample every 15 seconds)
    sub_wlen = int(sub_w_minutes * 4)
    for _, ts in df.groupby(level=['category', 'host']):
        ctx_wnds = list(mit.windowed(ts, wlen, sub_wlen))
        activities_wnd = map(lambda x: random_sublist(x, sub_wlen), ctx_wnds)
        coherency_tuples = map(lambda v: coherency_generator(*v, ctx_wnds), enumerate(ctx_wnds))
        # adding samples
        h_samples = zip(ctx_wnds, activities_wnd, coherency_tuples)
        for activity, ctx, (coherency_ctx, coherency_label) in h_samples:
            X.append({
                "activity": activity, 
                "context": ctx, 
                "coherency_ctx": coherency_ctx, 
                "coherency_label": coherency_label})

    def choice_n_merge(x):
        if x["coherency_ctxs"] is None:
            x["coherency_ctxs"] = random.choice(X)["context"] 
        merge = context_merge(x["context"], x["coherency_ctxs"])
        return (x["activity"], x["context"], merge, x["label"])
    X = map(choice_n_merge, X)

    return X


# ----- ----- LOSSES ----- ----- #
# ----- ----- ------ ----- ----- #
class Contextual_Coherency():
    def __init__(self, alpha=.5):
        self.alpha = alpha

    def __call__(self, x, _):
        activity, context, merged_context, merge_label = x
        context_loss = torch.norm(activity - context, 2)
        coherency_loss = F.binary_cross_entropy(merged_context, merge_label)
        ctx_coh = (self.alpha * context_loss) + ((1 - self.alpha) * coherency_loss)
        return ctx_coh 


# ----- ----- MODEL ----- ----- #
# ----- ----- ----- ----- ----- #
class Ts2Vec(torch.nn.Module):
    def __init__(self):
        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, h),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        edb = self.embedding(x)
        edb_norm = F.normalize(edb, p=2, dim=1)
        coh_score = self.coherency(edb_norm)
        return edb_norm, coh_score
