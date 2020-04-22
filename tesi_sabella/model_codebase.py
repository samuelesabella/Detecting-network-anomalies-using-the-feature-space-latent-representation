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


def coherency_generator(x, idx, time_windows, wlen, step_size):
    r = random.random()
    if r < .25:
        return (x, COH_TENSOR)
    if r < .5:
        x_b = time_windows[idx + (wlen / step_size)]
        return (x_b, COH_TENSOR)
    if r < .75:
        random_shift = random.randint(1, len(time_windows))
        x_b = time_windows[idx + (wlen / step_size) + random_shift]
        return (x_b, INC_TENSOR)
    return (None, INC_TENSOR)

def random_sublist(l, sub_wlen):
    r = random.randint(0, len(l)-sub_wlen)
    return l[r:r+sub_wlen]
    
def merge_b(x_a, x_b):
    r = zero_one_normal()
    r_idx = int(len(x_a) * r)
    return (x_a[:r_idx] + x_b[r_idx:])


def ts_windowing(df, w_minutes=15, sub_w_minutes=7, overlap=.75):
    X = []
    wlen = int(w_minutes * 4) # Samples per minutes (one sample every 15 seconds)
    sub_wlen = int(sub_w_minutes * 4)
    step_size = max(int(wlen * (1-overlap)), 1)
    for _, ts in df.groupby(level=['category', 'host']):
        wnds = list(mit.windowed(ts, wlen, step_size))
        sub_wnds = map(lambda x: random_sublist(x, sub_wlen), wnds)
        coherency_tuples = list(map(lambda v: coherency_generator(*v, wnds, wlen, step_size), enumerate(wnds)))
        coherency_wnds, coherency_label = zip(*coherency_tuples)
        X.append(sub_wnds, wnds, coherency_wnds, coherency_label)

    def choice_n_merge(x):
        if x[2] is None:
            x[2] = random.choice(X)[1] 
        return (x[0], x[1], merge_b(x[1], x[2]))
    X = map(choice_n_merge, X)

    return X


# ----- ----- LOSSES ----- ----- #
# ----- ----- ------ ----- ----- #
class Contextual_Coherency():
    def __init__(self, alpha=.5):
        self.alpha = alpha

    def __call__(self, x, _):
        sub_wnds, wnds, merged_coh, coh_label = x
        l_ctx = torch.norm(sub_wnds - wnds, 2)
        l_coh = F.binary_cross_entropy(merged_coh, coh_label)
        ctx_coh = (alpha * l_ctx) + ((1 - self.alpha) * l_coh)


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
