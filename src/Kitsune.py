import pandas as pd
import numpy as np
import KitNET as kit


def to_kitsune_input(samples):
    df = pd.concat(samples)
    channels = [ c for c in df.columns if (c[0]!="_") ]
    data = df[channels].values
    # np.random.shuffle(data)
    return data


class KitsuneAnomalyDetector():
    def __init__(self, maxAE, beta=1., tr_perc=.75, fm_perc=.1):
        """tr-perc will be used to compute phi
        """
        self.maxAE = maxAE
        self.beta = beta
        self.fm_perc = .1
        self.K = None
        self.tr_perc = tr_perc
        self.phi = -1

    def fit(self, df):
        kdata = to_kitsune_input(df)

        N = kdata.shape[0]
        channels = kdata.shape[1]

        tr_idx = int(self.tr_perc * N)
        kdata_tr = kdata[:tr_idx]
        N_tr = kdata_tr.shape[0]
        kdata_vl = kdata[tr_idx:] 
        N_vl = kdata_vl.shape[0]
        
        # Fitting ensemble ----- #
        FMgrace = int(N_tr * self.fm_perc) # instances taken to learn the feature mapping
        ADgrace = int(N_tr * (1-self.fm_perc)) # instances used to train the anomaly detector
        self.K = kit.KitNET(channels, self.maxAE, FMgrace, ADgrace-1)
        for i in range(N_tr):
            self.K.process(kdata[i, ])

        # Fitting phi ----- #
        vl_RMSEs = np.stack([ self.K.process(kdata_vl[i, ]) for i in range(N_vl) ])   
        self.phi = max(vl_RMSEs)
    
    def isanomaly(self, df, beta=None):
        if beta:
            self.beta = beta

        kdata = to_kitsune_input(df)
        N = kdata.shape[0]
        RMSEs = np.stack([ self.K.process(kdata[i, ]) for i in range(N) ])

        y_hat = (RMSEs > (self.phi * self.beta))
        y_hat = np.where(y_hat==True, 1, y_hat) 
        y_hat = np.where(y_hat==False, 0, y_hat) 

        df = pd.concat(df)
        df["_y_hat"] = y_hat
        return df



