import argparse
from pprint import pprint
import datetime
import glob
import numpy as np
from collections import defaultdict
from functools import partial
import influxdb
import requests


# ----- ----- PY_FLUX ----- ----- #
# ----- ----- ------- ----- ----- #
class FluxQueryFrom():
    def __init__(self, bucket):
        self.query = [f'from(bucket:"{bucket}")']

    def range(self, start='-24h', stop='now()'):
        self.query.append(f'range(start:{start}, stop:{stop})')
        return self

    def filter(self, fn):
        self.query.append(f'filter(fn:{fn})')
        return self

    def group(self, columns, mode='by'):
        cstr = str(columns).replace("\'", "\"")
        self.query.append(f'group(columns:{cstr}, mode:"{mode}")')
        return self

    def distinct(self, column):
        self.query.append(f'distinct(column:"{column}")')
        return self

    def __str__(self):
        return " |> ".join(self.query)


class Flux():
    def __init__(self, host='localhost', port=8086):
        self.session = requests.Session()
        url = f'http://{host}:{port}/api/v2/query'
        head = {
            'accept': 'application/csv',
            'content-type': 'application/vnd.flux'
        }
        self.preq = requests.Request('POST', url, headers=head)

    def __call__(self, q):
        self.preq.data = q
        res = self.session.send(self.preq.prepare())

        return res

    def show_tag_values(self, bucket, from_measurement, with_key,
                        trange='-24h'):
        q = FluxQueryFrom(bucket).range(trange).filter(
                '(r) => r._measurement == "host:traffic"').group([with_key]).distinct(with_key)

        return self(str(q))

    def show_measurements(self, bucket, trange='-24h'):
        q = FluxQueryFrom(bucket).range(trange).group(["_measurement"]).distinct("_measurement")

        return self(str(q))


# ----- ----- HOST DATA GENERATOR ----- ----- #
# ----- ----- ------------------- ----- ----- #
class InfluxHostDataGenerator():
    def __init__(self, t_conf, t_cumulative=False):
        self.last_timestamp = 0
        self.cumulative = t_cumulative
        self.history = defaultdict(partial(np.ndarray, 0))

        self.dbclient = influxdb.InfluxDBClient(
            host='localhost', port=t_conf['port'])
        self.dbclient.switch_database(t_conf['db_name'])

    def __getitem__(self, key):
        if not self.cumulative:
            raise ValueError(
                'No historical data available, cumulative flag set to False')
        return self.history[key]

    def batch_generator(self, t_cat, t_batch_size):
        if isinstance(t_cat, list):
            metrics = list(zip(*[self.history[c] for c in t_cat]))
        else:
            metrics = self.history[t_cat]

        for i in range(0, len(metrics), t_batch_size):
            yield metrics[i, i+t_batch_size]

    def category_shape(self):
        if not self.cumulative:
            raise ValueError('Cumulative flag set to False')
        return np.sum({k: v.shape for k, v in self.history.items()})

    def poll(self):
        diff_ts = {}
        q = self.query(self.last_timestamp)
        host_measurements = self.dbclient.query(q, epoch='ns').raw["series"]
        for m in host_measurements:
            hostname = m["tags"]["host"]
            category = self.category_map(hostname)
            if not category:
                continue

            v = np.array([x[1:] for x in m["values"]])
            if category not in diff_ts:
                diff_ts[category] = v
            else:
                diff_ts[category] = np.vstack([diff_ts[category], v])

            # Timestamp update
            last_t = m["values"][-1][0]
            if last_t > self.last_timestamp:
                self.last_timestamp = last_t

        if self.cumulative:
            for c, v in diff_ts.items():
                past_v = self.history[c]
                self.history[c] = np.vstack([past_v, v]) if past_v.size else v
        return diff_ts

    def save(self, dname=None):
        if not dname:
            dname = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        elif dname.endswith('/'):
            dname = dname[:-1]

        with open(f'{dname}/query.txt', 'w+') as f:
            f.write(self.query(12345))

        for k, v in self.history.items():
            np.save(f'{dname}/{k}.npy', v)

    def load(self, dname):
        if dname.endswith('/'):
            dname = dname[:-1]

        with open(f'{dname}/query.txt') as f:
            if f.read() != self.query(12345):
                raise ValueError('Trying to load from different query')

        self.history = {}
        for f in glob.glob(f'{dname}/*.npy'):
            k = f[:-4]
            v = np.fromfile(f'{dname}/{f}')
            self.history[k] = v

    def query(self, last_ts):
        """Returns an influxdb query over the metrics
        computed over multiple hosts
        """
        raise NotImplementedError

    def category_map(self, host):
        """Return the category, given a host
        """
        raise NotImplementedError


class TrafficDataGenerator(InfluxHostDataGenerator):
    def category_map(self, _):
        return 'pc-generic'

    def query(self, last_ts):
        """Bytes sent and received by a host since last query
        """
        return ('SELECT "bytes_rcvd", "bytes_sent" '
                'FROM "host:traffic" '
                f'WHERE time > {last_ts} '
                'GROUP BY "host"'
                'ORDER BY time ASC')


class CICIDS2017(TrafficDataGenerator):
    def __init__(self, *args, **kwargs):
        super(CICIDS2017, self).__init__(*args, **kwargs)
        self.ipv4_netmap = {
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
        self.mac_netmap = {
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
    generator = TrafficDataGenerator(DBCONF, WSIZE)
    generator.poll()
    print(generator.category_len())
