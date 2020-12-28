from collections import defaultdict
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
from fbprophet import Prophet


GYEAR = 0
def add_year(s):
    global GYEAR
    GYEAR += 1
    return s + pd.DateOffset(days=365*GYEAR)


def to_prophet_input(samples):
    samples = copy.deepcopy(samples)

    for df in samples:
        df.reset_index(inplace=True)
        for h in df["_host"].unique():
            host_mask = (df["_host"] == h)
            time_shifted = add_year(df.loc[host_mask, "_time"])
            df.loc[host_mask, "_time"] = time_shifted
    df = pd.concat(samples)
    df = df.rename(columns={"_time": "ds"})
    channels = [c for c in df.columns if ((c[0]!="_") and ("time:" not in c)) ]
    return df[channels]


class ProphetStorm():
    def __init__(self):
        self.P = defaultdict(lambda: Prophet(yearly_seasonality=True, 
                                             daily_seasonality=True, 
                                             weekly_seasonality=True))
    def fit(self, df):
        df = to_prophet_input(df)
        cols = set(df.columns)
        cols.remove("ds")
        
        for y in tqdm(cols):
            other_cols = set(cols)
            other_cols.remove(y)
            ts = df.rename(columns={y: "y"})
            m = self.P[y]
            for c in other_cols:
                m.add_regressor(c)
            m.fit(ts)
    
    def isanomaly(self, df):
        df = to_prophet_input(df)

        cols = set(df.columns)
        cols.remove("ds")
        attribute_scores = []
        
        for c in tqdm(cols):
            ground_truth = df[c]
            model_input = df.drop(c, axis=1)
            pred = self.P[c].predict(model_input)[["trend", "yhat", "yhat_lower", "yhat_upper"]].copy()
            pred["ground_truth"] = ground_truth.values

            out_of_forchetta = ((pred["yhat"] < pred["yhat_lower"]) | (pred["yhat"] > pred["yhat_upper"]))
            # Percentage of outliers in validation data for column 'c'
            attribute_scores.append(out_of_forchetta)
        
        # point_score = 1 - len(out_of_forchetta[out_of_forchetta==False]) / len(out_of_forchetta)
        y_hat = np.array(attribute_scores)
        y_hat = 1 - (np.sum(y_hat.astype(int), axis=0) / len(y_hat))
        df["_y_hat"] = y_hat
        return df
