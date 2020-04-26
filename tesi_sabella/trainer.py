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
    train, test = cb.data_split(model_samples, SEED)
    model_train = cb.ts_windowing(train)
    model_test = cb.ts_windowing(test)
    
    
    net = NeuralNet(
        cb.Ts2Vec, 
        cb.Contextual_Coherency,
        optimizer=torch.optim.Adam)
    
    params = {
        'lr': [0.01, 0.02],
        'max_epochs': [10, 20],
    }
    gs = GridSearchCV(net, params, refit=False, cv=5, scoring=["precision", "recall"])
    gs_results = gs.fit(model_samples)
