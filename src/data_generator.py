from collections import defaultdict
from sklearn import preprocessing
import signal
import influxdb_client
from influxdb_client import InfluxDBClient
from datetime import datetime
from sklearn.preprocessing import KBinsDiscretizer
import argparse
import ntopng_constants as ntopng_c
import numpy as np
import pandas as pd
import pathlib 
import logging
import pyfluxc.pyfluxc as flux
from pyts.approximation import SymbolicAggregateApproximation as SAX
import re
import requests 
import sys
import time

import importlib
importlib.reload(flux)


# ----- ----- PREPROCESSING ----- ----- #
# ----- ----- ------------- ----- ----- #
MORNING =   np.array([1, 0, 0, 0])
AFTERNOON = np.array([0, 1, 0, 0])
EVENING   = np.array([0, 0, 1, 0])
NIGHT     = np.array([0, 0, 0, 1])
hour2ts = [(MORNING,   range(6,  12)),
            (AFTERNOON, range(12, 17)),
            (EVENING,   range(17, 22)),
            (NIGHT,     list(range(22, 24))+list(range(0, 6)))]
hour2ts = { h: t for t, hrange in hour2ts for h in hrange }

class Preprocessor():
    def __init__(self, deltas=True, discretize=True, flevel="NF_BLMISC"):
        self.flevel = flevel 
        self.compute_deltas = deltas
        self.compute_discrtz = discretize
        self.discretizer = KBinsDiscretizer(n_bins=15, encode="ordinal") 

    @staticmethod
    def date_as_feature(df):
        time_indx = df.index.get_level_values("_time")
    
        weekend_map = defaultdict(lambda: 0, { 5: 1, 6: 1 })
        df["time:is_weekend"] = time_indx.dayofweek.map(weekend_map)
        
        mapped_hours = time_indx.hour.map(hour2ts).values.tolist() 
        hours_df = pd.DataFrame(mapped_hours, columns=["time:morning", "time:afternoon", "time:evening", "time:night"], index=df.index)
        df = pd.concat([df, hours_df], axis=1)
        return df
    
    @staticmethod
    def fillzero(df):
        """Replace zero traffic holes with rolling window mean
        """
        missing_traffic = (df == 0).all(axis=1)
        df[missing_traffic].replace(0, np.NaN)
        r_mean = df.rolling(min_periods=1, window=3, center=True).sum().shift(-1) / 2
        df.loc[missing_traffic] = r_mean[missing_traffic]
        return df
    
    def discretize(self, df, fit=False):
        tc = [c for c in df.columns if (("ndpi_flows:num_flows" not in c) and ("time:" not in c))]
        values = df[tc].values
        if fit:
            df[tc] = self.discretizer.fit_transform(values)
        else:
            df[tc] = self.discretizer.transform(values)
        return df

    def preprocessing(self, df):
        smart_features = set(ntopng_c.FEATURE_LEVELS[self.flevel])
        available_features = set(df.columns)
        available_cols = available_features.intersection(smart_features)
        if available_cols != smart_features:
            missing_c = smart_features - available_cols
            logging.warning(f"Missing columns: {missing_c}")
        df = df[available_cols].copy(deep=True)
        df[df<0] = 0
        df = df.fillna(0)
        df = Preprocessor.fillzero(df)
    
        # DPI unit length normalization ..... #
        ndpi_num_flows_c = [c for c in df.columns if "ndpi_flows:num_flows" in c]
        ndpi = df[ndpi_num_flows_c]
        ndpi_sum = ndpi.sum(axis=1)
        df.loc[:, ndpi_num_flows_c] = ndpi.divide(ndpi_sum + 1e-3, axis=0)        
    
        # Non decreasing delta discretization ..... #
        if self.compute_deltas:
            # Filter selected non-stationary features
            non_decreasing = [c for c in df.columns if c in ntopng_c.NON_DECREASING]
            df[non_decreasing] = df[non_decreasing].groupby(level=["device_category", "host"], group_keys=False).apply(lambda group: group.diff())
            df = df.fillna(0)
        
        # Feature discretization ..... #
        # Note: we avoided using pandas qcut/cut due to the decoupling between fit and transform 
        #       offered by scikit. In the future {KBinsDiscretizer} will be fitted once a week or so
        #       with weekly data and used multiple times while the model is running
        # if self.compute_discrtz:
        #     df = self.discretize(df, fit)

        # Date/hour as a feature ..... #
        df = Preprocessor.date_as_feature(df)
        
        return df


# ----- ----- HOST DATA GENERATOR ----- ----- #
# ----- ----- ------------------- ----- ----- #
class FluxDataGenerator():
    def __init__(self, bucket, windowsize, fluxclient, start, ntopng_conf):
        self.last_timestamp = start
        self.samples = None
        self.fluxclient = fluxclient
        self.bucket = bucket
        self.ntopng_conf = ntopng_conf
        self.host_map = {}
        self.windowsize = windowsize
        wndsize_val, wndsize_unit = re.match(r'([0-9]+)([a-zA-Z]+)', self.windowsize).groups() 
        self.window_timedelta = np.timedelta64(wndsize_val, wndsize_unit)

    def to_pandas(self):
        if self.samples is None:
            raise ValueError('No samples available')
        return self.samples.copy(deep=True)

    def pull(self, start=None, stop=None):
        if not start:
            start = self.last_timestamp
        utcnow = pd.Timestamp.utcnow()
        q = self.query(start, stop)

        client = InfluxDBClient(url="http://localhost:8086", token="my-token", org="my-org")
        query_api = client.query_api()
        try:
            query_reply = query_api.query_data_frame(str(q))
        except influxdb_client.rest.ApiException as e:
            if e != b'{"error":"failed to initialize execute state: no database"}\n':
                logging.warning(e)
            self.last_timestamp = utcnow if stop is None else stop
            return self.last_timestamp, None 
        except Exception as e: 
            logging.warning("Exception: probably database not ready")
            return self.last_timestamp, None
        new_samples = pd.concat(query_reply) if type(query_reply)==list else query_reply
        
        if new_samples.empty:
            self.last_timestamp = utcnow if stop is None else stop
            return self.last_timestamp, None 
        new_samples = new_samples.drop(columns=["result", "table"])

        # Transforming existing ndpi flows to measurements ..... #
        host_ndpi_flows_measurements = new_samples["_measurement"]=="host:ndpi_flows"
        host_ndpi_flows = new_samples.loc[host_ndpi_flows_measurements]
        host_ndpi_flows_cat = host_ndpi_flows["protocol"].str.lower().map(ntopng_c.NDPI_VALUE2CAT)
        new_samples.loc[host_ndpi_flows_measurements, "_field"] += ("__" + host_ndpi_flows_cat)
        # Transforming existing ndpi bytes to measurements ..... #
        host_ndpi_bytes_measurements = new_samples["_measurement"]=="host:ndpi"
        host_ndpi_bytes = new_samples.loc[host_ndpi_bytes_measurements]
        host_ndpi_bytes_cat = host_ndpi_bytes["protocol"].str.lower().map(ntopng_c.NDPI_VALUE2CAT)
        new_samples.loc[host_ndpi_bytes_measurements, "_field"] += ("__" + host_ndpi_bytes_cat)
        # Device category ..... #
        new_samples['device_category'] = self.category_map(new_samples)
        # Building dframe ..... # 
        new_samples['_key'] = new_samples['_measurement'].str.replace('host:', '') + ':' + new_samples['_field']
        new_samples = new_samples.pivot_table(index=["device_category", "host", "_time"], 
                                              columns="_key", values="_value", aggfunc=np.sum)
        new_samples.columns = new_samples.columns.rename(None)
        # Drop cutted samples. E.g. range(start=13:46:58, stop:13:49:00) have almost for sure NaN in the first 2 seconds) 
        # Thus we drop NaN values from bytes_rcvd which should never be NaN
        new_samples.dropna(subset=["traffic:bytes_rcvd"])
        # Adding missing columns ..... #
        missing_columns = []
        available_columns = set(new_samples.columns)
        missing_columns += ntopng_c.NDPI_FLOWS_COMPLETE - available_columns 
        missing_columns += ntopng_c.NDPI_BYTES_RCVD_COMPLETE - available_columns 
        missing_columns += ntopng_c.NDPI_BYTES_SENT_COMPLETE - available_columns 
        new_samples = new_samples.reindex(columns=new_samples.columns.tolist() + missing_columns, fill_value=0)
        # Updating ..... #
        # Checking to have only valid columns
        new_samples = new_samples[ntopng_c.FEATURES_COMPLETE]
        # Removing duplicate timestamps
        if self.samples is not None:
            max_old_samples = self.samples.groupby(level=["device_category", "host"]).apply(lambda x: x.index.max())
            min_new_samples = new_samples.groupby(level=["device_category", "host"]).apply(lambda x: x.index.min())
            dup_samples = np.intersect1d(max_old_samples, min_new_samples)
            new_samples = new_samples.drop(dup_samples)
        # Merging and updating time
        self.samples = pd.concat([self.samples, new_samples]) 
        self.last_timestamp = new_samples.index.get_level_values("_time").max()

        return self.last_timestamp, new_samples

    def save(self, datapath:pathlib.Path = None):
        if not datapath:
            datapath = datetime.now().strftime("%m.%d.%Y_%H.%M.%S_data")
            datapath = pathlib.Path(datapath)

        # Storing a generic query
        with open(datapath / 'query.txt', 'w+') as f:
            f.write(self.query(12345))
        
        self.samples.to_pickle(datapath / 'timeseries.pkl')

    def load(self, dname:pathlib.Path):
        with open(dname / 'query.txt') as f:
            if f.read() != self.query(12345):
                raise ValueError('Trying to load from different query')
        self.samples = pd.read_pickle(dname / 'timeseries.pkl')

    def query(self, start, stop=None):
        q = flux.FluxQueryFrom(self.bucket)
        q.range(start=start, stop=stop)
        q.filter('(r) => r._measurement =~ /host:.*/')
        q.aggregateWindow(every=self.windowsize, fn='mean')
        q.drop(columns=['_start', '_stop', 'ifid'])
        q.group(columns=["_time", "host"])
        return q

    def category_map(self, new_samples):
        return pd.Series(["unknown"]*len(new_samples))
        # unique_hosts = new_samples["host"].unique()
        # unique_unknonw = [x for x in unique_hosts if x not in self.host_map]

        # ntopng_host, ntopng_port = self.conf["ntopng"] 
        # ntopng_user, ntopng_passwd = self.conf["ntopng_credentials"]
        # url = f"https://{ntopng_host}:{ntopng_port}/lua/rest/get/host/data.lua?host="
        # params = { "cookie": f"user={ntopng_user}; password={ntopng_passwd}" } 
        # for h in unique_unknonw:
        #     r = requests.get(url = url + h, params = params) 
        #     # TODO: extract class
        #     raise NotImplementedError("TODO")
        #     data = r.json()
        # return new_samples["host"].map(self.host_map)


class SigCatcher:
    def __init__(self, signals):
        self.received = None
        for s in signals:
            signal.signal(signal.SIGINT, self.catcher)

    def catcher(self, signum, frame):
        self.received = signum


# ----- ----- CLI ----- ----- #
# ----- ----- --- ----- ----- #
if __name__ == "__main__":
    # Parser definition ..... #
    print("packet2ts")
    parser = argparse.ArgumentParser(description="packet2ts")
    parser.add_argument("-b", "--bucket", help="ntopng flux-bucket name", default="ntopng")
    parser.add_argument("--influxport", help="influxdb port", type=int, default=8086)
    parser.add_argument("--ntopngport", help="ntopng port", type=int, default=3000)
    parser.add_argument("--credentials", help="ntopng REST API credential 'user:password'", type=str, default="admin:admin")
    parser.add_argument("-e", "--every", help="sample data every X minutes", type=int, default=15)
    parser.add_argument("-o", "--output", help="output dataframe filename")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.bucket}__{datetime.now().strftime('%m.%d.%Y_%H.%M.%S')}.pkl"

    ntopng_credentials = args.credentials.split(":")
    if len(ntopng_credentials) != 2:
        print("Error, wrong credential format")
        sys.exit(0)
    ntopng_conf = {
        "ntopng": ("localhost", args.ntopngport),
        "credentials": ntopng_credentials
    }
    
    fclient = flux.FluxClient(host="127.0.0.1", port=args.influxport); 
    start = pd.Timestamp.utcnow() - pd.DateOffset(minutes=args.every)
    generator = FluxDataGenerator(args.bucket, "15s", fclient, start, ntopng_conf)

    sigcatch = SigCatcher([signal.SIGINT, signal.SIGTERM])
    while sigcatch.received is None:
        generator.pull()
        print(f"Polled at: {datetime.now().strftime('%m.%d.%Y_%H.%M.%S')}")
        time.sleep(60 * args.every)
    print("Closing capture")
    generator.pull()
    df = generator.to_pandas()
    df.to_pickle(f"{args.output}.pkl")
