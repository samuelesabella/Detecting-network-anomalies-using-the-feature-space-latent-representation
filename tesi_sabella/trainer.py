import torch
import pandas as pd
import argparse
import random
import numpy as np
import more_itertools as mit
from skorch.net import NeuralNet
from sklearn.model_selection import GridSearchCV

import model_codebase as cb
import cicids2017_data_generator as cicids2017


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
    df = df[(df.index.get_level_values("_time").day == 3) & (df.index.get_level_values("_time").hour < 12)]
    pr = cicids2017.Cicids2017Preprocessor()
    df = pr.preprocessing(df, update=True)
    
    # train_groups, test_groups = cb.data_split(model_samples, SEED)

    model_samples = cb.ts_windowing(df)
    X_train = np.concatenate([model_samples["activity"], 
                              model_samples["coherent_activity"], 
                              model_samples["context"]], axis=1)
    Y_train = model_samples["coherency_label"]
    
    net = NeuralNet(
        cb.Ts2Vec, 
        cb.Contextual_Coherency,
        optimizer=torch.optim.Adam)
    
    params = {
        'lr': [0.01, 0.02],
        'max_epochs': [10, 20],
    }
    gs = GridSearchCV(net, params, refit=False, cv=5, scoring=["precision", "recall"])
    gs_results = gs.fit(X_train, Y_train)
