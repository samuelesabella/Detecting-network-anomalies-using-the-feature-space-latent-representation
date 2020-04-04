import pandas as pd
import unittest
from unittest import mock
from unittest.mock import patch
import os
import sys
sys.path.append(f"{os.path.dirname(__file__)}/../tesi_sabella")
from tesi_sabella.data_generator import FluxDataGenerator 


# ----- ----- CONSTANTS ----- ----- #
# ----- ----- --------- ----- ----- #
MOCK_FEATURE_SET = [
    "traffic:bytes_rcvd",
    "host_unreachable_flows:flows_as_client"]
MOCK_L4_BYTES_RCVD_COMPLETE = set([ "l4protos:bytes_rcvd__icmp" ])
MOCK_L4_BYTES_SENT_COMPLETE = set([ "l4protos:bytes_sent__icmp" ])

MOCK_NDPI_FLOWS_COMPLETE = set([
    "ndpi_flows:num_flows__gaming",
    "ndpi_flows:num_flows__p2p_file_sharing"])
MOCK_NDPI_BYTES_RCVD_COMPLETE = set([
    "ndpi:bytes_rcvd__gaming",
    "ndpi:bytes_rcvd__p2p_file_sharing"])
MOCK_NDPI_BYTES_SENT_COMPLETE = set([
    "ndpi:bytes_sent__gaming",
    "ndpi:bytes_sent__p2p_file_sharing"])
MOCK_NDPI_VALUE2CAT = {
    'gnutella': 'p2p_file_sharing',
    'warcraft3': 'gaming'}
    

# ----- ----- DUMMY VALUES ----- ----- #
# ----- ----- ----------- ----- ----- #
mock_qresult = pd.DataFrame([
    ("192.168.10.1", 0.0, "2020-04-03 12:25:15", None,   None,       "bytes_rcvd",      "host:traffic"),
    ("192.168.10.1", 10., "2020-04-03 12:25:15", None,   None,       "flows_as_client", "host:host_unreachable_flows"),
    ("192.168.10.1", 20., "2020-04-03 12:25:15", "icmp", None,       "bytes_rcvd",      "host:l4protos"),
    ("192.168.10.1", 40., "2020-04-03 12:25:15", None,   "gnutella",  "num_flows",      "host:ndpi_flows"),
    ("192.168.10.1", 50., "2020-04-03 12:25:15", None,   "warcraft3", "num_flows",      "host:ndpi_flows"),
    ("192.168.10.1", 51., "2020-04-03 12:25:15", None,   "warcraft3", "num_flows",      "host:ndpi_flows"),
    
    ("192.168.10.1", 1.0, "2020-04-03 12:25:30", None,   None,        "bytes_rcvd",      "host:traffic"),
    ("192.168.10.1", 11., "2020-04-03 12:25:30", None,   None,        "flows_as_client", "host:host_unreachable_flows"),
    ("192.168.10.1", 21., "2020-04-03 12:25:15", "icmp", None,        "bytes_rcvd",      "host:l4protos"),
    ("192.168.10.1", 41., "2020-04-03 12:25:15", None,   "gnutella",  "num_flows",       "host:ndpi_flows"),
    ("192.168.10.1", 51., "2020-04-03 12:25:15", None,   "warcraft3", "num_flows",       "host:ndpi_flows"),
    ("192.168.10.1", 52., "2020-04-03 12:25:15", None,   "warcraft3", "num_flows",       "host:ndpi_flows"),

    ("192.168.10.1", 2.0, "2020-04-03 12:25:45", None,   None,        "bytes_rcvd",      "host:traffic"),
    ("192.168.10.1", 12., "2020-04-03 12:25:45", None,   None,        "flows_as_client", "host:host_unreachable_flows"),
    ("192.168.10.1", 22., "2020-04-03 12:25:15", "icmp", None,        "bytes_rcvd",      "host:l4protos"),
    ("192.168.10.1", 42., "2020-04-03 12:25:15", None,   "gnutella",  "num_flows",       "host:ndpi_flows"),
    ("192.168.10.1", 52., "2020-04-03 12:25:15", None,   "warcraft3", "num_flows",       "host:ndpi_flows"),
    ("192.168.10.1", 53., "2020-04-03 12:25:15", None,   "warcraft3", "num_flows",       "host:ndpi_flows"),

    ("192.168.10.1", 3.0, "2020-04-03 12:26:00", None,   None,        "bytes_rcvd",      "host:traffic"),
    ("192.168.10.1", 13., "2020-04-03 12:26:00", None,   None,        "flows_as_client", "host:host_unreachable_flows"), 
    ("192.168.10.1", 23., "2020-04-03 12:25:15", "icmp", None,        "bytes_rcvd",      "host:l4protos"),
    ("192.168.10.1", 43., "2020-04-03 12:25:15", None,   "gnutella",  "num_flows",       "host:ndpi_flows"),
    ("192.168.10.1", 53., "2020-04-03 12:25:15", None,   "warcraft3", "num_flows",       "host:ndpi_flows"),
], columns=["host", "_value", "_time", "l4proto", "protocol", "_field", "_measurement"])
mock_qresult["_time"] = mock_qresult["_time"].apply(lambda s: pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S'))


# ----- ----- TESTING ----- ----- #
# ----- ----- ------- ----- ----- #
@patch("ntopng_constants.FEATURE_SET",              MOCK_FEATURE_SET)
@patch("ntopng_constants.L4_BYTES_RCVD_COMPLETE",   MOCK_L4_BYTES_RCVD_COMPLETE)
@patch("ntopng_constants.L4_BYTES_SENT_COMPLETE",   MOCK_L4_BYTES_SENT_COMPLETE)
@patch("ntopng_constants.NDPI_FLOWS_COMPLETE",      MOCK_NDPI_FLOWS_COMPLETE)
@patch("ntopng_constants.NDPI_BYTES_RCVD_COMPLETE", MOCK_NDPI_BYTES_RCVD_COMPLETE)
@patch("ntopng_constants.NDPI_BYTES_SENT_COMPLETE", MOCK_NDPI_BYTES_SENT_COMPLETE)
@patch("ntopng_constants.NDPI_VALUE2CAT",           MOCK_NDPI_VALUE2CAT)

class TestInfluxHostDataGenerator(unittest.TestCase):
    @mock.patch("pyfluxc.pyfluxc.FluxClient")
    def setUp(self, mock_influx_client):
        type(mock_influx_client.return_value).dframe = mock_qresult
        self.client = mock_influx_client
        self.cicids2017 = FluxDataGenerator("test_bucket", "15s", self.client, pd.Timestamp.utcnow())

    def check_basic(self):
        """Check basic features polling
        """
        _, sample = self.cicids2017.pull()
        target = pd.DataFrame({
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:15"): [0., 22.],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:30"): [1., 23.],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:45"): [2., 24.],  
            ("unknown device class", "192.168.10.1", "2020-04-03 12:26:00"): [3., 25.]
        }).T
        target.columns = ["active_flows:flow_as_client", "host_unreachable_flows:flow_as_client"]
        target["_time"] = target["_time"].apply(lambda s: pd.to_datetime(s, format='%Y-%m-%dT%H:%M:%SZ'))
        
        pd.testing.assert_frame_equal(sample[MOCK_FEATURE_SET], target)

    def check_l4(self):
        pass

    def check_ndpi(self):
        _, sample = self.cicids2017.pull()
        target = pd.DataFrame({
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:15"): [40., 50.5],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:30"): [41., 51.5],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:45"): [42., 52.5],  
            ("unknown device class", "192.168.10.1", "2020-04-03 12:26:00"): [43., 53.] 
        }).T
        target.columns = ["ndpi:bytes_rcvd__gaming", "ndpi:bytes_rcvd__p2p_file_sharing"]
        target["_time"] = target["_time"].apply(lambda s: pd.to_datetime(s, format='%Y-%m-%dT%H:%M:%SZ'))
        
        pd.testing.assert_frame_equal(sample[MOCK_NDPI_FLOWS_COMPLETE], target)
