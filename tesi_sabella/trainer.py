import torch
import pandas as pd
from collections import defaultdict
import argparse
import random
import numpy as np
from skorch.net import NeuralNet
from sklearn.model_selection import GridSearchCV

import model_codebase as cb
import cicids2017_data_generator as cicids2017

import logging
logging.basicConfig(level=logging.DEBUG)


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
    pr = cicids2017.Cicids2017Preprocessor()
    
    df_train = df[df.index.get_level_values("_time").day == 3]
    df_train = pr.preprocessing(df_train, update=True)
    train_set = cb.ts_windowing(df_train, overlapping=.8)
    train_tensors = cb.windows2tensor(train_set)

    test_set = defaultdict(list)
    for d in [4, 5, 6, 7]:
        df_day = df[df.index.get_level_values("_time").day == d]
        df_day_preproc = pr.preprocessing(df_day)
        test_day = cb.ts_windowing(df_day_preproc)
        
        attacks_rows = torch.where(test_day["attack"] == cb.ATTACK_TRAFFIC)[0].numpy()
        normal_rows = torch.where(test_day["attack"] == cb.NORMAL_TRAFFIC)[0].numpy()
        normal_rows = np.random.choice(normal_rows, len(attacks_rows), replace=False)

        sampled_rows = np.concatenate([normal_rows, attacks_rows])
        sample_df = test_day["X"].loc[sampled_rows]
        sample_coh_label = test_day["coherency_label"][sampled_rows]
        sample_attack_label = test_day["attack"][sampled_rows]

        test_set["X"].append(sample_df)
        test_set["coherency_label"].append(sample_coh_label)
        test_set["attack"].append(sample_attack_label)

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
