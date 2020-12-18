import pandas as pd
from collections import defaultdict
from skorch.callbacks import EpochScoring
import matplotlib.pyplot as plt
import numpy as np
import skorch
import AnomalyDetector as ad
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
        X_tr, _ = skorch.utils.data_from_dataset(dataset_train)
        X_vl, _ = skorch.utils.data_from_dataset(dataset_valid)
        self.plot_dist(net, X_tr, "train")
        self.plot_dist(net, X_vl, "valid")


class DetectionScore(skorch.callbacks.EpochScoring):
    def __init__(self, measure, dt: pd.DataFrame):
        self.measure = measure

        dt = dt.copy()
        dt["_y"] = dt["_isanomaly"]
        dt.loc[dt["_isanomaly"] == "none", "_y"] = ad.NORMAL_TRAFFIC
        dt.loc[dt["_isanomaly"] != "none", "_y"] = ad.ATTACK_TRAFFIC

        self.data = []
        days = np.unique(dt.index.get_level_values("_time").day)
        for d in days:
            daily_df = dt[dt.index.get_level_values("_time").day == d]
            self.data.append(daily_df)
        
        score_name = f"DT_{self.__name__}"
        super(DetectionScore, self).__init__(self.score, on_train=False, lower_is_better=False, name=score_name)

    @property
    def __name__(self):
        return f"{self.measure.__name__}"

    def score(self, net, *args, **kwargs):
        df = net.pointwise_anomaly(self.data)
        y = np.concatenate([x["_y"] for x in df])
        y_hat = np.concatenate([x["_y_hat"] for x in df])
         
        res = self.measure(y.astype(float), np.round(y_hat))
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

