import argparse
import datetime
import pandas as pd
import tesi_sabella.pyflux as flux
import os
import pathlib 


# ----- ----- HOST DATA GENERATOR ----- ----- #
# ----- ----- ------------------- ----- ----- #
class FluxDataGenerator():
    def __init__(self, fluxclient):
        self.last_timestamp = '-48h'
        self.samples = None
        self.fluxclient = fluxclient

    def toPandas(self):
        return self.samples

    def poll(self):
        q = self.query(self.last_timestamp)
        new_samples = self.fluxclient(q, grouby=False).dframe
        new_samples['device_category'] = new_samples.apply(self.category_map, axis=1)
        new_samples = new_samples.dropna(subset=['device_category']) 
        self.samples = pd.concat([self.samples, new_samples])
        self.last_timestamp = max(new_samples['_time'])
        return self.last_timestamp

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

    def query(self, last_ts):
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


class CICIDS2017_Generator(FluxDataGenerator):
    def __init__(self, bucket, *args, **kwargs):
        super(CICIDS2017_Generator, self).__init__(*args, **kwargs)
        self.bucket = bucket

    def query(self, last_ts):
        q = flux.FluxQueryFrom(self.bucket)
        q.range(start=last_ts)
        q.filter('(r) => r._measurement == "host:traffic" '
                 'or r._measurement == "host:udp_pkts" '
                 'or r._measurement == "dns_qry_rcvd_rsp_sent" ')
        q.aggregateWindow(every='1h', fn='mean')
        q.drop(columns=['_start', '_stop', 'ifid'])
        q.group(columns=["_time", "host"])
        return q

    def category_map(self, qres_row):
        hostname = qres_row['host']
        if hostname in CICIDS2017_IPV4_NETMAP:
            return CICIDS2017_IPV4_NETMAP[hostname]
        if hostname in CICIDS2017_MAC_NETMAP:
            return CICIDS2017_MAC_NETMAP[hostname]
        return None


# ----- ----- CLI ----- ----- #
# ----- ----- --- ----- ----- #
if __name__ == '__main__':
    # Parser definition ..... #
    print('packet2ts')
    parser = argparse.ArgumentParser(description='packet2ts')
    parser.add_argument('-d', '--database',
                        help='ntop influx database name', default='ntopng')
    parser.add_argument('-p', '--port', help='influxdb port',
                        type=int, default=8086)

    # Parser call ..... #
    args = parser.parse_args()
    fclient = flux.Flux(port=args.port)
    generator = CICIDS2017_Generator('CICIDS2017_Monday_from15to16/autogen', fclient)
    generator.poll()
