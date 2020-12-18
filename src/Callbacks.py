from collections import defaultdict
import itertools
import pandas as pd
from scipy.stats import truncnorm
from sklearn import preprocessing
from sklearn.manifold import TSNE
from skorch.callbacks import EpochScoring
from tqdm import tqdm
import matplotlib.pyplot as plt
import more_itertools as mit
import numpy as np
import skorch
import torch


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


class Ts2VecScore():
    def __init__(self, measure, data=None):
        self.measure = measure
        self.data = data

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
        data = dset if self.data is None else self.data

        with torch.no_grad():
            y_hat = net.module_.context_anomaly(data.X["context"])
        res = self.measure(data.y.cpu(), np.round(y_hat.cpu()))
        return res


class AnchorPlot(skorch.callbacks.Callback):
    def __init__(self, path):
        self.path = path
        self.flabel = "distance"

    @property
    def __name__(self):
        return "distance_visualization"

    def on_train_begin(self, *_, **__):
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
        X_tr, _ = skorch.utils.data_from_dataset(dataset_train)
        X_vl, _ = skorch.utils.data_from_dataset(dataset_valid)
        self.plot_dist(net, X_tr, "train")
        self.plot_dist(net, X_vl, "valid")

