import argparse
import datetime
import pandas as pd
import tesi_sabella.pyFluxClient.pyfluxclient as flux
import pathlib 


# ----- ----- HOST DATA GENERATOR ----- ----- #
# ----- ----- ------------------- ----- ----- #
class FluxDataGenerator():
    def __init__(self, fluxclient, start):
        self.last_timestamp = start
        self.samples = None
        self.fluxclient = fluxclient

    def toPandas(self):
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


ntophost_measurements = [
    "host:udp_pkts", "host:active_flows",
    "host:contacts", "host:dns_qry_rcvd_rsp_sent",
    "host:dns_qry_sent_rsp_rcvd", "host:echo_packets",
    "host:echo_reply_packets", "host:engaged_alerts",
    "host:host_unreachable_flows", "host:l4protos",
    "host:misbehaving_flows", "host:score",
    "host:tcp_packets", "host:tcp_rx_stats",
    "host:tcp_tx_stats", "host:total_alerts",
    "host:total_flow_alerts", "host:total_flows",
    "host:traffic", "host:udp_pkts",
    "host:udp_sent_unicast", "host:unreachable_flows",
    "host:ndpi", "host:ndpi_flows"]

class ntop_Generator(FluxDataGenerator):
    def __init__(self, bucket, windowsize, *args, **kwargs):
        super(ntop_Generator, self).__init__(*args, **kwargs)
        self.bucket = bucket
        self.windowsize = windowsize

    def query(self, start, stop=None):
        q = flux.FluxQueryFrom(self.bucket)
        q.range(start=start, stop=stop)

        # filtering ..... #
        host_filter = '(r) => ' 
        measurement_or = ''.join([f'or r._measurement == "{x}" ' for x in ntophost_measurements]) 
        host_filter += measurement_or[3:]
        q.filter(host_filter)

        q.aggregateWindow(every=self.windowsize, fn='mean')
        q.drop(columns=['_start', '_stop', 'ifid'])
        q.group(columns=["_time", "host"])
        return q

    def toPandas(self):
        self.samples['_key'] = self.samples['_measurement'].str.replace('host:', '') + ':' + self.samples['_field']
        groups =  ['device_category', 'host', '_time', '_key']
        return self.samples.groupby(groups)['_value'].sum(min_count=1).unstack('_key')

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
    fclient = flux.FluxClient(port=args.port)
    generator = CICIDS2017_Generator('CICIDS2017_Monday_from15to16/autogen', fclient)
    generator.poll()
