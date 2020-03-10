import argparse
import datetime
import pandas as pd
import tesi_sabella.pyflux as flux
import pathlib 


# ----- ----- HOST DATA GENERATOR ----- ----- #
# ----- ----- ------------------- ----- ----- #
class FluxDataGenerator():
    def __init__(self, fluxclient):
        self.last_timestamp = 0
        self.history = {}
        self.category_ts = {}
        self.fluxclient = fluxclient

    def category_split(self):
        for h, ts in self.history:
            cat = self.category_map(h)
            if cat in self.category_ts:
                self.category_ts[cat] = pd.cat([self.category_ts, ts])
            else:
                self.category_ts[cat] = ts
        self.history = {}

    def poll(self):
        q = self.query(self.last_timestamp)
        new_samples = self.fluxclient(q).dframe

        for ((h, _), dframe) in new_samples:
            if not self.category_map(h):
                continue

            if h in self.history:
                self.history = pd.cat([self.history[h], dframe])
            else:
                self.history[h] = dframe

        self.last_timestamp = max([x[0][1] for x in new_samples])

    def save(self, datapath:pathlib.Path = None):
        if not datapath:
            datapath = datetime.now().strftime("%m.%d.%Y_%H.%M.%S_data")
            datapath = pathlib.Path(datapath)

        # Storing a generic query
        with open(datapath / 'query.txt', 'w+') as f:
            f.write(self.query(12345))
        
        for h, dframe_ts in self.history.items():
            host_path = datapath / f'hosts/{h}.pkl'
            host_path = str(host_path.absolute()) 
            dframe_ts.to_pickle(host_path)
        
        for h, dframe_ts in self.category_ts.items():
            cat_path = datapath / f'category_ts/{h}.pkl'
            cat_path = str(cat_path.absolute()) 
            dframe_ts.to_pickle(cat_path)


    def load(self, dname):
        if dname.endswith('/'):
            dname = dname[:-1]

        with open(f'{dname}/query.txt') as f:
            if f.read() != self.query(12345):
                raise ValueError('Trying to load from different query')

        # TODO: implement file loading
        

    def query(self, last_ts):
        """Returns an iterable of tuple <(host, timestamp), samples>
        with {samples} a dataframe of tuple <measurement, value>
        """
        raise NotImplementedError

    def category_map(self, host):
        """Return the category, given a host
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


class CICIDS2017(FluxDataGenerator):
    def query(self, last_ts):
        q = flux.FluxQueryFrom('CICIDS2017_Monday_from15to16/autogen')
        q.range(start=last_ts)
        q.filter('(r) => r._measurement == "host:traffic" '
                 'or r._measurement == "host:udp_pkts" '
                 'or r._measurement == "dns_qry_rcvd_rsp_sent" ')
        q.aggregateWindow(every='1h', fn='mean')
        q.drop(columns=['_start', '_stop', 'ifid'])
        q.group(columns=["_time", "host"])

    def category_map(self, hostname):
        if hostname in self.ipv4_netmap:
            return self.ipv4_netmap[hostname]
        elif hostname in self.mac_netmap:
            return self.mac_netmap[hostname]
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
    WSIZE = args.wsize
    DBCONF = {"port": args.port, "db_name": args.database}
    pcap = args.pcap

    # Populating influxdb using ntop and the given pcap ..... #

    # Logic ..... #
    generator = FluxDataGenerator(DBCONF, WSIZE)
    generator.poll()
    print(generator.category_len())
