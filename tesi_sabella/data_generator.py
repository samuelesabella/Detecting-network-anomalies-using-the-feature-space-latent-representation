import argparse
import re
import numpy as np
from datetime import datetime
import time
import pandas as pd
import pyfluxc.pyfluxc as flux
import pathlib 
import ntopng_constants as ntopng_c

import importlib
importlib.reload(flux)



# ----- ----- CICIDS2017 ----- ----- #
# ----- ----- ---------- ----- ----- #
CICIDS2017_IPV4_NETMAP = {
    "192.168.10.3": "server",
    "192.168.10.50": "server",
    "192.168.10.51": "server",
    "205.174.165.68": "server",
    "205.174.165.66": "server", 

    "192.168.10.19": "pc",
    "192.168.10.17": "pc",
    "192.168.10.16": "pc",
    "192.168.10.12": "pc",
    "192.168.10.9": "pc",
    "192.168.10.5": "pc",
    "192.168.10.8": "pc",
    "192.168.10.14": "pc",
    "192.168.10.15": "pc",
    "192.168.10.25": "pc",
}

CICIDS2017_MAC_NETMAP = {
    "18:66:DA:9B:E3:7D": "server",
    "00:19:B9:0A:69:F1": "server",
    "B8:AC:6F:36:0B:A8": "server",

    "00:23:AE:9B:AD:B3": "pc",
    "00:23:AE:9B:95:67": "pc",
    "00:23:AE:9B:8A:BF": "pc",
    "B8:AC:6F:36:04:E3": "pc",
    "B8:AC:6F:1D:1F:6C": "pc",
    "B8:AC:6F:36:0A:8B": "pc",
    "B8:AC:6F:36:08:F5": "pc",
    "B8:AC:6F:36:07:EE": "pc",
    "00:1E:4F:D4:CA:28": "pc",
    "00:25:00:A8:C4:60": "pc",
}


# ----- ----- HOST DATA GENERATOR ----- ----- #
# ----- ----- ------------------- ----- ----- #
class FluxDataGenerator():
    def __init__(self, bucket, windowsize, fluxclient, start):
        self.last_timestamp = start
        self.samples = None
        self.fluxclient = fluxclient
        self.bucket = bucket
        self.windowsize = windowsize
        wndsize_val, wndsize_unit = re.match(r'([0-9]+)([a-zA-Z]+)', self.windowsize).groups() 
        self.window_timedelta = np.timedelta64(wndsize_val, wndsize_unit)

    def to_pandas(self):
        if self.samples is None:
            raise ValueError('No samples available')
        
        no_diff_cols = ["active_flows:flows_as_client", "active_flows:flows_as_server"]
        backup_cols = self.samples[no_diff_cols]
        delta_samples = self.samples.drop(no_diff_cols, axis=1).groupby(level=1).diff()
        delta_samples[no_diff_cols] = backup_cols
        delta_samples = delta_samples.groupby(level=1).apply(lambda group: group.iloc[1:])
        return delta_samples

    def pull(self, start=None, stop=None):
        if not start:
            start = self.last_timestamp
        utcnow = pd.Timestamp.utcnow()
        q = self.query(start, stop)
        new_samples = self.fluxclient(q, grouby=False).dframe
        if new_samples is None:
            self.last_timestamp = utcnow
            return None

        # Transforming existing ndpi flows to measurements ..... #
        host_ndpi_flows = new_samples.loc[new_samples["_measurement"]=="host:ndpi_flows"]
        host_ndpi_flows_cat = host_ndpi_flows["protocol"].map(ntopng_c.NDPI_VALUE2CAT)
        new_samples.loc[host_ndpi_flows.index, "_field"] += ("__" + host_ndpi_flows_cat)
        # Transforming existing ndpi bytes to measurements ..... #
        host_ndpi_bytes = new_samples.loc[new_samples["_measurement"]=="host:ndpi"]
        host_ndpi_bytes_cat = host_ndpi_bytes["protocol"].map(ntopng_c.NDPI_VALUE2CAT)
        new_samples.loc[host_ndpi_bytes.index, "_field"] += ("__" + host_ndpi_bytes_cat)
        # Transforming existing l4-proto to measurements ..... #
        host_l4protos = new_samples.loc[new_samples["_measurement"]=="host:l4protos"]
        new_samples.loc[host_l4protos.index, "_field"] += ("__" + host_l4protos["l4proto"])
        # Device category ..... #
        new_samples['device_category'] = new_samples.apply(self.category_map, axis=1)
        # Building dframe ..... # 
        new_samples['_key'] = new_samples['_measurement'].str.replace('host:', '') + ':' + new_samples['_field']
        new_samples = new_samples.pivot_table(index=["device_category", "host", "_time"], 
                                              columns="_key", values="_value", aggfunc="mean")
        new_samples = new_samples[ntopng_c.FEATURE_SET]
        # Drop cutted samples. E.g. range(start=13:46:58, stop:13:49:00) have almost for sure NaN in the first 2 seconds) 
        # Thus we drop NaN values from bytes_rcvd which should never be NaN
        new_samples.dropna(subset=["traffic:bytes_rcvd"])
        # ndpi and l4 protocols have often NaN
        new_samples = new_samples.fillna(0)
        # Adding missing columns ..... #
        missing_columns = []
        available_columns = set(new_samples.columns)
        missing_columns += ntopng_c.NDPI_FLOWS_COMPLETE - available_columns 
        missing_columns += ntopng_c.NDPI_BYTES_RCVD_COMPLETE - available_columns 
        missing_columns += ntopng_c.NDPI_BYTES_SENT_COMPLETE - available_columns 
        missing_columns += ntopng_c.L4_BYTES_RCVD_COMPLETE - available_columns 
        missing_columns += ntopng_c.L4_BYTES_SENT_COMPLETE - available_columns 
        new_samples = new_samples.reindex(columns=new_samples.columns.tolist() + missing_columns, fill_value=0)
        # Updating ..... #
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

    def category_map(self, qres_row):
        hostname = qres_row['host']
        if hostname in CICIDS2017_IPV4_NETMAP:
            return CICIDS2017_IPV4_NETMAP[hostname]
        if hostname in CICIDS2017_MAC_NETMAP:
            return CICIDS2017_MAC_NETMAP[hostname]
        return "unknown device class"

    def check_jumps(self, new_samples):
        if self.samples is None or new_samples is None:
            return
        last_timestamp = {}
        one_second = np.timedelta64(1000000000, 'ns')

        g = new_samples.groupby(['device_category', 'host'])
        for (_, host), host_samples in g:
            s_times = host_samples.index.get_level_values(2).sort_values(ascending=True)
            s_times = s_times.sort_values(ascending=True)
            last_timestamp[host] = s_times[0]
            # intra-sample jump check ..... #
            delta = np.diff(s_times.values)
            blind_spot = [(*s_times[i:i+2], x_delta) for i, x_delta in enumerate(delta)]
            blind_spot = filter(lambda x: x[2] > self.window_timedelta, blind_spot)
            blind_spot = [(*x[:2], x[2] / one_second) for x in blind_spot]
            blind_spot = pd.DataFrame(blind_spot, columns=["start", "stop", "seconds"])
            if len(blind_spot) > 0:
                raise RuntimeError(f"Intra blind spots for {host}: \n {blind_spot}")

        # inter-sample check ..... #    
        g = self.samples.groupby(['device_category', 'host'])
        for (_, host), host_samples in g:
            if host not in last_timestamp:
                continue
            old_s_times = host_samples.index.get_level_values(2).sort_values(ascending=True)
            old_s_times = old_s_times.sort_values(ascending=True)
            last_old_ts = old_s_times[-1]
            if (last_old_ts - last_timestamp[host]).to_numpy() > self.window_timedelta:
                raise RuntimeError(f"Inter-poll blind spot for host: {host}")


# ----- ----- CLI ----- ----- #
# ----- ----- --- ----- ----- #
if __name__ == '__main__':
    # Parser definition ..... #
    print('packet2ts')
    parser = argparse.ArgumentParser(description='packet2ts')
    parser.add_argument('-b', '--bucket',
                        help='ntop influx database name', default='ntopng')
    parser.add_argument('-p', '--port', help='influxdb port',
                        type=int, default=8086)
    parser.add_argument('-e', '--every', help="poll every x minutes",
                        type=int, default=15)
    parser.add_argument('-o', '--output', help="output file name")
    args = parser.parse_args()
    if args.output is None:
        args.output = f"{args.bucket}__{datetime.now().strftime('%m.%d.%Y_%H.%M.%S')}.pkl"
    
    fclient = flux.FluxClient(host='127.0.0.1', port=args.port); 
    start = pd.Timestamp.utcnow() - pd.DateOffset(minutes=args.every)
    cicids2017 = FluxDataGenerator(args.bucket, '15s', fclient, start)

    while True:
        try:
            cicids2017.pull()
            df = cicids2017.to_pandas()
            print(f'Polled at: {datetime.now().strftime("%m.%d.%Y_%H.%M.%S")}')
            time.sleep(60 * args.every)
        except KeyboardInterrupt:
            print('Closing capture')
            break
