import argparse
import re
import numpy as np
from datetime import datetime
import time
import pandas as pd
import pyfluxc.pyfluxc as flux
import pathlib 
import requests 
import ntopng_constants as ntopng_c
import sys

import importlib
importlib.reload(flux)


# ----- ----- PREPROCESSING ----- ----- #
# ----- ----- ------------- ----- ----- #
def fill_zero_traffic(df):
    """Replace zero traffic holes with rolling window mean
    """
    traffic = ["traffic:bytes_rcvd", "traffic:bytes_sent"]
    missing_traffic = (df[traffic] == 0).all(axis=1)
    df[missing_traffic].replace(0, np.NaN)
    r_mean = df[traffic].rolling(min_periods=2, window=3, center=True).sum() / 2
    df.loc[missing_traffic, traffic] = r_mean[missing_traffic]
    return df

def preprocessing(df):
    df = df[ntopng_c.FEATURE_LEVELS["smart"]].copy(deep=True)
    df = fill_zero_traffic(df)

    # DPI unit length normalization ..... #
    ndpi_num_flows_c = [c for c in df.columns if "ndpi_flows:num_flows" in c]
    ndpi = df[ndpi_num_flows_c]
    ndpi_sum = ndpi.sum(axis=1)
    df.loc[:, ndpi_num_flows_c] = ndpi.divide(ndpi_sum, axis=0)        

    # Non decreasing delta discretization ..... #
    non_decreasing = ["traffic:bytes_rcvd", "traffic:bytes_sent"]
    df[non_decreasing] = df[non_decreasing].diff()
    df = df.groupby(level=["device_category", "host"], group_keys=False).apply(lambda group: group.iloc[1:])

    # Min max scaling for others ..... #
    other_cols = [c for c in df.columns if c not in ndpi_num_flows_c and c not in non_decreasing]
    other_cols_min = df[other_cols].min()
    other_cols_max = df[other_cols].max()
    df.loc[:, other_cols] = (df.loc[:, other_cols] - other_cols_min) / (other_cols_max - other_cols_min)

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

    #@staticmethod
    #def fix_jumps(df):
    #    host_groups = df.groupby(['device_category', 'host'])
    #    for (_, h), host_samples in host_groups:
    #        times = host_samples.index.sort_values(ascending=True)
    #        delta = times.to_series().diff()[1:]
    #        delta_gap = filter(lambda x: x[1] > window_timedelta, enumerate(delta))

    @staticmethod
    def fix_zero_traffic(df):
        """Replace zero traffic holes with zero mean
        """
        traffic = ["traffic:bytes_rcvd", "traffic:bytes_sent"]

        zero_traffic = (df[traffic] == 0).all(axis=1)
        index_hours = df.index.get_level_values("_time").hour
        
        missing_traffic = zero_traffic & (index_hours > 8) & (index_hours < 17)
        df[missing_traffic].replace(0, np.NaN)
        r_mean = df[traffic].rolling(min_periods=2, window=3, center=True).sum() / 2
        df[missing_traffic] = r_mean[missing_traffic]
        return df

    @staticmethod
    def pre_processing(df):
        df = df[ntopng_c.FEATURE_LEVELS["smart"]].copy(deep=True)
        df = self.fix_zero_traffic(df)

        # DPI unit length normalization ..... #
        ndpi_num_flows_c = [c for c in df.columns if "ndpi_flows:num_flows" in c]
        ndpi = df[ndpi_num_flows_c]
        ndpi_sum = ndpi.sum(axis=1)
        df.loc[:, ndpi_num_flows_c] = ndpi.divide(ndpi_sum, axis=0)        

        # Non decreasing ..... #
        non_decreasing = ["traffic:bytes_rcvd", "traffic:bytes_sent"]
        df[non_decreasing] = df[non_decreasing].diff()
        df = df.groupby(level=["device_category", "host"], group_keys=False).apply(lambda group: group.iloc[1:])

        # Min max scaling ..... #
        other_cols = [c for c in df.columns if c not in ndpi_num_flows_c and c not in non_decreasing]
        other_cols_min = df[other_cols].min()
        other_cols_max = df[other_cols].max()
        df.loc[:, other_cols] = (df.loc[:, other_cols] - other_cols_min) / (other_cols_max - other_cols_min)

        return df

    def pull(self, start=None, stop=None):
        if not start:
            start = self.last_timestamp
        utcnow = pd.Timestamp.utcnow()
        q = self.query(start, stop)
        new_samples = self.fluxclient(q, grouby=False).dframe
        if new_samples is None:
            self.last_timestamp = utcnow if stop is None else stop
            return self.last_timestamp, None 

        # Transforming existing ndpi flows to measurements ..... #
        host_ndpi_flows = new_samples.loc[new_samples["_measurement"]=="host:ndpi_flows"]
        host_ndpi_flows_cat = host_ndpi_flows["protocol"].str.lower().map(ntopng_c.NDPI_VALUE2CAT)
        new_samples.loc[host_ndpi_flows.index, "_field"] += ("__" + host_ndpi_flows_cat)
        # Transforming existing ndpi bytes to measurements ..... #
        host_ndpi_bytes = new_samples.loc[new_samples["_measurement"]=="host:ndpi"]
        host_ndpi_bytes_cat = host_ndpi_bytes["protocol"].str.lower().map(ntopng_c.NDPI_VALUE2CAT)
        new_samples.loc[host_ndpi_bytes.index, "_field"] += ("__" + host_ndpi_bytes_cat)
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
        # ndpi protocols have often NaN
        # new_samples = new_samples.fillna(0)
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
        self.samples = pd.concat([self.samples, new_samples])
        cat_host_sample = new_samples.index[0][:2]
        samples_times = new_samples.loc[cat_host_sample].index 
        self.last_timestamp = max(samples_times)

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
        unique_hosts = new_samples.unique()
        unique_unknonw = [x for x in unique_hosts if x not in self.host_map]

        ntopng_host, ntopng_port = self.conf["ntopng"] 
        ntopng_user, ntopng_passwd = self.conf["ntopng_credentials"]
        url = f"https://{ntopng_host}:{ntopng_port}/lua/rest/get/host/data.lua?host="
        params = { "cookie": f"user={ntopng_user}; password={ntopng_passwd}" } 
        for h in unique_unknonw:
            r = requests.get(url = url + h, params = params) 
            # TODO: extract class
            raise NotImplementedError("TODO")
            data = r.json()
        return new_samples["host"].map(self.host_map)


# ----- ----- CLI ----- ----- #
# ----- ----- --- ----- ----- #
if __name__ == "__main__":
    # Parser definition ..... #
    print("packet2ts")
    parser = argparse.ArgumentParser(description="packet2ts")
    parser.add_argument("-b", "--bucket", help="ntopng flux-bucket name", default="ntopng")
    parser.add_argument("--influxport", help="influxdb port", type=int, default=8086)
    parser.add_argument("--ntopngport", help="ntopng port", type=int, default=3000)
    parser.add_argument("--credentials", help="ntopng REST API credential 'user:password'", type=str)
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
    
    fclient = flux.FluxClient(host="127.0.0.1", port=args.port); 
    start = pd.Timestamp.utcnow() - pd.DateOffset(minutes=args.every)
    generator = FluxDataGenerator(args.bucket, "15s", fclient, start, ntopng_conf)

    while True:
        try:
            generator.pull()
            print(f"Polled at: {datetime.now().strftime('%m.%d.%Y_%H.%M.%S')}")
            time.sleep(60 * args.every)
        except KeyboardInterrupt:
            print("Closing capture")
            df = generator.to_pandas()
            df.to_pickle(f"{args.output}.pkl")
            break
