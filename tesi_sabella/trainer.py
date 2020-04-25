import torch
import pandas as pd
import argparse
import random
import numpy as np
import more_itertools as mit
from skorch.net import NeuralNet
from sklearn.model_selection import GridSearchCV

import model_codebase as cb
import cicids2017_data_generator as generator


# Reproducibility .... #
SEED = 117697 
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mEmbedding training")
    parser.add_argument("--datasetpath", "-d", help="Dataset path")
    args = parser.parse_args()
    
    df = pd.read_pickle(args.datasetpath)
    df = generator.preprocessing(df)
    model_samples = cb.ts_windowing(df)
    
    membedder_net = cb.mEmbeddingNet()
    net = NeuralNet(
        cb.mEmbeddingNet, 
        cb.TripletLoss(),
        optimizer=torch.optim.Adam)
    
    params = {
        'lr': [0.01, 0.02],
        'max_epochs': [10, 20],
        'module__num_units': [10, 20],
        'criterion__difficulty': ['hard']
    }
    gs = GridSearchCV(net, params, refit=False, cv=5)
    gs_results = gs.fit(tr_x)
