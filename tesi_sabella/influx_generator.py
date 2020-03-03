import argparse
import datetime
import glob
import numpy as np
from collections import defaultdict
from functools import partial
import influxdb 


# ----- ----- HOST DATA GENERATOR ----- ----- #
# ----- ----- ------------------- ----- ----- #
class InfluxHostDataGenerator():
    def __init__(self, t_conf, t_cumulative=False):
        self.last_timestamp = 0 
        self.cumulative = t_cumulative
        self.history = defaultdict(partial(np.ndarray, 0))

        self.dbclient = influxdb.InfluxDBClient(host='localhost', port=t_conf['port'])
        self.dbclient.switch_database(t_conf['db_name'])
    
    def __getitem__(self, key):
        if not self.cumulative:
            raise ValueError('No historical data available, cumulative flag set to False')
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
        host_ts = {} 
        q = self.query(self.last_timestamp)
        host_measurements = self.dbclient.query(q, epoch='ns').raw["series"]
        for m in host_measurements:
            hostname = m["tags"]["host"] 
            category = self.category_map(hostname)
            v = np.array([x[1:] for x in m["values"]])
            host_ts[category] = v
            
            # Timestamp update
            last_t = m["values"][-1][0] 
            if last_t > self.last_timestamp:
                self.last_timestamp = last_t
             
            if self.cumulative:
                past_v = self.history[category]
                self.history[category] = np.vstack([past_v, v]) if past_v.size else v
    
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


# ----- ----- CLI ----- ----- #
# ----- ----- --- ----- ----- #
if __name__ == '__main__':
    # Parser definition ..... #
    print('packet2ts')
    parser = argparse.ArgumentParser(description='packet2ts')
    parser.add_argument('pcap', help='Target pcap file', nargs='?')
    parser.add_argument('-w', '--wsize', help='window size', type=int, default=1024)
    parser.add_argument('-d', '--database', help='ntop influx database name', default='ntopng')
    parser.add_argument('-p', '--port', help='influxdb port', type=int, default=8086)

    # Parser call ..... #
    args = parser.parse_args()
    WSIZE = args.wsize
    DBCONF = { "port": args.port, "db_name": args.database }
    pcap = args.pcap

    # Populating influxdb using ntop and the given pcap ..... #

    # Logic ..... #
    generator = TrafficDataGenerator(DBCONF, WSIZE)
    generator.poll()
    print(generator.category_len())
    
