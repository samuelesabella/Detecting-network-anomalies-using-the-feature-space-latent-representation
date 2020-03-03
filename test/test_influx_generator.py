import numpy as np
import copy
import unittest
from unittest import mock
from tesi_sabella.influx_generator import TrafficDataGenerator


# ----- ----- QUERY TEMPLATES ----- ----- #
# ----- ----- --------------- ----- ----- #
query_traffic_host_tmpl = {
    'columns': ['time', 'bytes_rcvd', 'bytes_sent'],
    'name': 'host:traffic',
    'tags': {'host': 'HOSTNAME_HERE'},
    'values': [
        ['2020-02-28T10:52:15Z', 81, 180],
        ['2020-02-28T10:52:45Z', 82, 181],
        ['2020-02-28T10:53:15Z', 83, 182]]
}
query_traffic_tmpl = { 
    'series': [query_traffic_host_tmpl], 
    'statement_id': 0 
}

def traffic_influx_tmpl(hosts, timeseries):
    res = copy.deepcopy(query_traffic_host_tmpl)
    res['series'] = []
    for h, ts in zip(hosts, timeseries):
        t = copy.deepcopy(query_traffic_host_tmpl)
        t['tags']['host'] = h
        t['values'] = ts
        res['series'].append(t)
    return res


# ----- ----- DATA GENERATOR ----- ----- #
# ----- ----- -------------- ----- ----- #
class TestInfluxHostDataGenerator(unittest.TestCase):
    @mock.patch('influxdb.InfluxDBClient.__init__')
    def setUp(self, mock_influx_client):
        # Object patch ..... #
        mock_influx_client.return_value = None
        # Object initialization ..... #
        self.windowsize = 15
        self.db_configuration = {"db_name": "fake-db", "port": 0000}
        self.generator = TrafficDataGenerator(self.db_configuration, self.windowsize)

    @mock.patch('influxdb.InfluxDBClient.query')
    def test_sample_host(self, mock_query):
        # Influx measurement ..... #
        h1 = '1.2.3.4'
        h1_ts = [
            ['2020-02-28T10:52:15Z', 81, 180], 
            ['2020-02-28T10:52:45Z', 82, 170], 
            ['2020-02-28T10:53:15Z', 83, 190]]
        h2 = '172.0.0.1'
        h2_ts = [
            ['2020-02-28T10:52:15Z', 10, 1], 
            ['2020-02-28T10:52:45Z', 10, 2], 
            ['2020-02-28T10:53:15Z', 10, 3]]
        ts = np.array([x[1:] for x in (h1_ts+h2_ts)])

        # Patch attribute ..... #
        fake_db_result = traffic_influx_tmpl([h1, h2], [h1_ts, h2_ts])
        type(mock_query.return_value).raw = fake_db_result

        sample = self.generator.poll()['generic-pc']
        self.assertEqual(sample, ts, 'incorrect sample')
