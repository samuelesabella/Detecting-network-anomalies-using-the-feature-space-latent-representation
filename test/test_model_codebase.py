import pandas as pd
import numpy as np
import torch
import unittest
import copy
import pathlib
import random

import os
import sys
sys.path.append(f"{os.path.dirname(__file__)}/../tesi_sabella")

import model_codebase as cb
import tesi_sabella.cicids2017 as cicids2017


# Reproducibility .... #
SEED = 117697 
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


TIMESERIES_PATH = pathlib.Path("./dataset/CICIDS2017_complete.pkl")

pr = cicids2017.Cicids2017Preprocessor()
df = pd.read_pickle(TIMESERIES_PATH)

df_monday = df[df.index.get_level_values("_time").day == 3]
df_monday = pr.preprocessing(df_monday, update=True)
samples = cb.ts_windowing(df_monday, overlapping=.85, pick_coherent_activity=False)


# ----- ----- DUMMY VALUES ----- ----- #
# ----- ----- ----------- ----- ----- #
class TestModelCodebase(unittest.TestCase):
    def setUp(self):
        self.samples = copy.deepcopy(samples)

    @staticmethod
    def is_coherent(df):
        ctx = df.xs("context", level="model_input")["_time"]
        coh_ctx = df.xs("coherency_activity", level="model_input")["_time"]

        diff = (coh_ctx - ctx).values
        return (diff == np.timedelta64(7, "m")).all() or (diff == np.timedelta64(0)).all()

    @staticmethod
    def is_incoherent(df):
        ctx = df.xs("context", level="model_input")["_time"]
        coh_ctx = df.xs("coherency_activity", level="model_input")["_time"]
        diff = ((coh_ctx - ctx) if coh_ctx.max() > ctx.max() else (ctx - coh_ctx)).values
        all_distant = (diff >= np.timedelta64(1, "h")).all()

        ctx_host = df.xs("context", level="model_input")["host"].iloc[0]
        coh_host = df.xs("coherency_activity", level="model_input")["host"].iloc[0]
        
        res = (all_distant or (ctx_host != coh_host))
        return res


    def test_coherency(self):
        coh_idx = torch.where(self.samples["coherency"] == cb.COHERENT)[0].unique()
        coh_idx = np.random.choice(coh_idx, 1000, replace=False)

        coh_tuples = self.samples["X"].loc[coh_idx]
        coherency_test = coh_tuples.groupby(level="sample_idx").apply(self.is_coherent)
        assert(coherency_test.all())

        incoh_idx = torch.where(self.samples["coherency"] == cb.INCOHERENT)[0].unique().tolist()
        incoh_idx = np.random.choice(incoh_idx, 1000, replace=False)

        incoh_tuples = self.samples["X"].loc[incoh_idx]
        incoherent_test = incoh_tuples.groupby(level="sample_idx").apply(self.is_incoherent)
        assert(incoherent_test.all())


    # def test_random_sublist(self):
