import argparse
import re
import numpy as np
import logging
import signal
from datetime import datetime
import time
import pandas as pd
import pyfluxc as flux
import pathlib 


# ----- ----- HOST DATA GENERATOR ----- ----- #
# ----- ----- ------------------- ----- ----- #
class FluxDataGenerator():
    def __init__(self, fluxclient, start):
        self.last_timestamp = start
        self.samples = None
        self.fluxclient = fluxclient

    def to_pandas(self):
        return self.samples

    def poll(self, start=None, stop=None):
        if not start:
            start = self.last_timestamp
        q = self.query(start, stop)
        new_samples = self.fluxclient(q, grouby=False).dframe
        if new_samples is None:
            return None
        new_samples['device_category'] = new_samples.apply(self.category_map, axis=1)
        new_samples = new_samples.dropna(subset=['device_category']) 
        self.samples = pd.concat([self.samples, new_samples])
        self.last_timestamp = max(new_samples['_time'])

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
        """Returns an iterable of tuple <(host, timestamp), samples>
        with {samples} a dataframe of tuple <measurement, value>
        """
        raise NotImplementedError

    def category_map(self, qres_row):
        """Returns the category of a device given a sample
        from a flux query result. Devices with {None} category are ignored
        """
        raise NotImplementedError


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


class ntop_Generator(FluxDataGenerator):
    def __init__(self, bucket, windowsize, *args, **kwargs):
        super(ntop_Generator, self).__init__(*args, **kwargs)
        self.bucket = bucket
        self.windowsize = windowsize

    def query(self, start, stop=None):
        q = flux.FluxQueryFrom(self.bucket)
        q.range(start=start, stop=stop)
        q.filter('(r) => r._measurement =~ /host:.*/')
        q.aggregateWindow(every=self.windowsize, fn='mean')
        q.drop(columns=['_start', '_stop', 'ifid'])
        q.group(columns=["_time", "host"])
        import pdb; pdb.set_trace() 
        return q

    def poll(self, **kwargs):
        wndsize_val, wndsize_unit = re.match(r'([0-9]+)([a-zA-Z]+)', '15s').groups() 

        last_timestamp, new_samples = super().poll(**kwargs)
        # Showing blind spots
        g = new_samples.groupby(['host', '_measurement', '_field'])
        for name, group in g:
            host = name[0]
            measurement = '->'.join(name[1:])
            times = group['_time'].sort_values(ascending=True)
            
            delta = np.diff(times.values)
            blind_spot = [(*times[i:i+2], x_delta) for i, x_delta in enumerate(delta)]
            blind_spot = filter(lambda x: x[2] > np.timedelta64(wndsize_val, wndsize_unit), blind_spot)
            one_second = np.timedelta64(1000000000, 'ns')
            blind_spot = [(*x[:2], x[2] / one_second) for x in blind_spot]
            blind_spot = pd.DataFrame(blind_spot, columns=["start", "stop", "seconds"])
            
            if len(blind_spot) > 0:
                logging.warning(f"Blind spots for {host}: {measurement}\n {blind_spot}")
        return last_timestamp

    def to_pandas(self):
        self.samples['_key'] = self.samples['_measurement'].str.replace('host:', '') + ':' + self.samples['_field']
        groups =  ['device_category', 'host', '_time', '_key']
        df = self.samples.groupby(groups)['_value'].sum(min_count=1).unstack('_key')
        # Cleaning the dataset
        df_clean = df.fillna({"score:score": 0})
        return df_clean.dropna()

    def category_map(self, qres_row):
        hostname = qres_row['host']
        if hostname in CICIDS2017_IPV4_NETMAP:
            return CICIDS2017_IPV4_NETMAP[hostname]
        if hostname in CICIDS2017_MAC_NETMAP:
            return CICIDS2017_MAC_NETMAP[hostname]
        return "unknown device class"


# ----- ----- CLI ----- ----- #
# ----- ----- --- ----- ----- #
if __name__ == '__main__':
    # Parser definition ..... #
    print('packet2ts')
    parser = argparse.ArgumentParser(description='packet2ts')
    parser.add_argument('-d', '--bucket',
                        help='ntop influx database name', default='ntopng')
    parser.add_argument('-p', '--port', help='influxdb port',
                        type=int, default=8086)
    parser.add_argument('-e', '--every', help="poll every x minutes",
                        type=int, default=15)
    parser.add_argument('-o', '--output', help="output file name")
    args = parser.parse_args()
    if args.output is None:
        args.output = f"{args.bucket}__{datetime.now().strftime("%m.%d.%Y_%H.%M.%S")}.pkl"
    
    fclient = flux.FluxClient(port=args.port); 
    start = pd.Timestamp.now() - pd.DateOffset(minutes=args.every)
    cicids2017 = ntop_Generator(args.bucket, '30s', fclient, start)

    running = True
    def signal_handler(*args):
        running = False
    signal.signal(signal.SIGINT, signal_handler)

    while running:
        cicids2017.poll()
        df = cicids2017.to_pandas()
        df.to_pickle(args.output)
        time.sleep(60 * args.every)
    
