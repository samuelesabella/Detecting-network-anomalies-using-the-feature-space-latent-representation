import matplotlib.pyplot as plt
from collections import defaultdict
import os
import numpy as np
import json
from datetime import datetime


class ResultLogger():
    def __init__(self, path='res'):
        if path.endswith('/'):
            path = path[:-1]
        t = datetime.now().strftime('%m.%d.%Y_%H.%M.%S')
        self.path = f'{path}/{t}'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        self.log_tree = defaultdict(list)
        self.tmp_batch = defaultdict(list)

    def log(self, metrics):
        for m, v in metrics.items():
            v = float(v)
            self.tmp_batch[m].append(v)

    def collapse(self, metrics):
        res = []
        if isinstance(metrics, str):
            metrics = [metrics]

        for m in metrics:
            meanv = np.mean(self.tmp_batch[m])
            res.append(meanv)
            self.log_tree[m].append(meanv)
            self.tmp_batch[m] = []
        return res 

    def dump_plot(self, keys, filename):
        for m in keys:
            plt.plot(self.log_tree[m], label=m)
        plt.legend()

        plt.savefig(f'{self.path}/{filename}.png')
        plt.cla()

    def dump(self, filename):
        with open(f'{self.path}/{filename}.log', 'w+') as outfile:
            json.dump(self.log_tree, outfile)

    def load(self, filename):
        with open(f'{self.path}/{filename}', 'r') as jsonfile:
            d = json.load(jsonfile)
        self.log_tree = defaultdict(list, d)
