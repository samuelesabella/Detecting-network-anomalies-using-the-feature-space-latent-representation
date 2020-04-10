import pandas as pd
import copy
import unittest
from unittest.mock import patch
import os
import sys
sys.path.append(f"{os.path.dirname(__file__)}/../tesi_sabella")
from tesi_sabella.data_generator import FluxDataGenerator 


# ----- ----- CONSTANTS ----- ----- #
# ----- ----- --------- ----- ----- #

MOCK_NDPI_FLOWS_COMPLETE = set([
    "ndpi_flows:num_flows__game",
    "ndpi_flows:num_flows__download-filetransfer-filesharing"])
MOCK_NDPI_BYTES_RCVD_COMPLETE = set([
    "ndpi:bytes_rcvd__game",
    "ndpi:bytes_rcvd__download-filetransfer-filesharing"])
MOCK_NDPI_BYTES_SENT_COMPLETE = set([
    "ndpi:bytes_sent__game",
    "ndpi:bytes_sent__download-filetransfer-filesharing"])
MOCK_NDPI_VALUE2CAT = {
    "gnutella": "download-filetransfer-filesharing",
    "warcraft3": "game",
    "quake": "game"}

MOCK_BASIC_FEATURES = set([
    "traffic:bytes_rcvd",
    "active_flows:flows_as_client",
    "active_flows:flows_as_server"])
MOCK_FEATURES_COMPLETE = copy.deepcopy(MOCK_BASIC_FEATURES)
MOCK_FEATURES_COMPLETE |= MOCK_NDPI_FLOWS_COMPLETE | MOCK_NDPI_BYTES_SENT_COMPLETE | MOCK_NDPI_FLOWS_COMPLETE


# ----- ----- DUMMY VALUES ----- ----- #
# ----- ----- ----------- ----- ----- #
mock_qresult = pd.DataFrame([
    ("192.168.10.1", 0.0,  "2020-04-03 12:25:15", None,   None,        "bytes_rcvd",      "host:traffic"),
    ("192.168.10.1", 10.,  "2020-04-03 12:25:15", None,   None,        "flows_as_client", "host:active_flows"),
    ("192.168.10.1", 10.5, "2020-04-03 12:25:15", None,   None,        "flows_as_server", "host:active_flows"),
    ("192.168.10.1", 40.0, "2020-04-03 12:25:15", None,   "gnutella",  "num_flows",      "host:ndpi_flows"),
    ("192.168.10.1", 50.,  "2020-04-03 12:25:15", None,   "warcraft3", "num_flows",      "host:ndpi_flows"),
    ("192.168.10.1", 51.,  "2020-04-03 12:25:15", None,   "quake",     "num_flows",      "host:ndpi_flows"),
    
    ("192.168.10.1", 1.0,  "2020-04-03 12:25:30", None,   None,        "bytes_rcvd",      "host:traffic"),
    ("192.168.10.1", 11.,  "2020-04-03 12:25:30", None,   None,        "flows_as_client", "host:active_flows"),
    ("192.168.10.1", 11.5, "2020-04-03 12:25:30", None,   None,        "flows_as_server", "host:active_flows"),
    ("192.168.10.1", 41.,  "2020-04-03 12:25:30", None,   "gnutella",  "num_flows",       "host:ndpi_flows"),
    ("192.168.10.1", 51.,  "2020-04-03 12:25:30", None,   "warcraft3", "num_flows",       "host:ndpi_flows"),
    ("192.168.10.1", 52.,  "2020-04-03 12:25:30", None,   "quake",     "num_flows",       "host:ndpi_flows"),

    ("192.168.10.1", 2.0,  "2020-04-03 12:25:45", None,   None,        "bytes_rcvd",      "host:traffic"),
    ("192.168.10.1", 12.,  "2020-04-03 12:25:45", None,   None,        "flows_as_client", "host:active_flows"),
    ("192.168.10.1", 12.5, "2020-04-03 12:25:45", None,   None,        "flows_as_server", "host:active_flows"),
    ("192.168.10.1", 42.1, "2020-04-03 12:25:45", None,   "gnutella",  "num_flows",       "host:ndpi_flows"),
    ("192.168.10.1", 52.,  "2020-04-03 12:25:45", None,   "warcraft3", "num_flows",       "host:ndpi_flows"),
    ("192.168.10.1", 53.,  "2020-04-03 12:25:45", None,   "quake",     "num_flows",       "host:ndpi_flows"),

    ("192.168.10.1", 3.0,  "2020-04-03 12:26:00", None,   None,        "bytes_rcvd",      "host:traffic"),
    ("192.168.10.1", 13.,  "2020-04-03 12:26:00", None,   None,        "flows_as_client", "host:active_flows"), 
    ("192.168.10.1", 13.5, "2020-04-03 12:26:00", None,   None,        "flows_as_server", "host:active_flows"),
    ("192.168.10.1", 43.3, "2020-04-03 12:26:00", None,   "gnutella",  "num_flows",       "host:ndpi_flows"),
    ("192.168.10.1", 53.,  "2020-04-03 12:26:00", None,   "warcraft3", "num_flows",       "host:ndpi_flows"),
], columns=["host", "_value", "_time", "l4proto", "protocol", "_field", "_measurement"])
mock_qresult["_time"] = mock_qresult["_time"].apply(lambda s: pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S"))


# ----- ----- TESTING ----- ----- #
# ----- ----- ------- ----- ----- #
@patch("ntopng_constants.FEATURES_COMPLETE",        MOCK_FEATURES_COMPLETE)
@patch("ntopng_constants.BASIC_FEATURES",           MOCK_BASIC_FEATURES)
@patch("ntopng_constants.NDPI_FLOWS_COMPLETE",      MOCK_NDPI_FLOWS_COMPLETE)
@patch("ntopng_constants.NDPI_BYTES_RCVD_COMPLETE", MOCK_NDPI_BYTES_RCVD_COMPLETE)
@patch("ntopng_constants.NDPI_BYTES_SENT_COMPLETE", MOCK_NDPI_BYTES_SENT_COMPLETE)
@patch("ntopng_constants.NDPI_VALUE2CAT",           MOCK_NDPI_VALUE2CAT)

class TestInfluxHostDataGenerator(unittest.TestCase):
    def setUp(self):
        self.patcher_influx_client = patch("pyfluxc.pyfluxc.FluxClient")
        self.client = self.patcher_influx_client.start()
        type(self.client.return_value).dframe = mock_qresult.copy(deep=True)
        self.cicids2017 = FluxDataGenerator("test_bucket", "15s", self.client, pd.Timestamp.utcnow())
    
    def tearDown(self):
        self.patcher_influx_client.stop()

    def test_basic(self):
        """Check basic features polling
        """
        _, sample = self.cicids2017.pull()
        target = pd.DataFrame({
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:15"): [0., 10.],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:30"): [1., 11.],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:45"): [2., 12.],  
            ("unknown device class", "192.168.10.1", "2020-04-03 12:26:00"): [3., 13.]
        }).T
        target.columns = ["traffic:bytes_rcvd", "active_flows:flows_as_client"]
        target.index = pd.MultiIndex.from_tuples([(dc, h, pd.to_datetime(d, format="%Y-%m-%d %H:%M:%S")) for (dc, h, d) in target.index])
        target.index = target.index.rename(["device_category", "host", "_time"])
        
        pd.testing.assert_frame_equal(sample[target.columns], target, check_like=True)

    def test_ndpi(self):
        _, sample = self.cicids2017.pull()
        target = pd.DataFrame({
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:15"): [40., 101.],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:30"): [41., 103.],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:45"): [42.1, 105.],  
            ("unknown device class", "192.168.10.1", "2020-04-03 12:26:00"): [43.3, 53.] 
        }).T
        target.columns = ["ndpi_flows:num_flows__download-filetransfer-filesharing", "ndpi_flows:num_flows__game"]
        target.index = pd.MultiIndex.from_tuples([(dc, h, pd.to_datetime(d, format="%Y-%m-%d %H:%M:%S")) for (dc, h, d) in target.index])
        target.index = target.index.rename(["device_category", "host", "_time"])
        
        pd.testing.assert_frame_equal(sample[MOCK_NDPI_FLOWS_COMPLETE], target, check_like=True)

    def test_deltas(self):
        _, i_sample = self.cicids2017.pull()
        ii_mock_qresult = mock_qresult.copy(deep=True)
        ii_mock_qresult["_value"] += 4.
        ii_mock_qresult["_time"] += pd.Timedelta(minutes=1) 
        type(self.client.return_value).dframe = ii_mock_qresult.copy(deep=True)
        _, ii_sample = self.cicids2017.pull()
        dset = self.cicids2017.to_pandas(delta=True)

        target = pd.DataFrame({
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:30"): [1., 1.0, 11.5],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:45"): [1., 1.1, 12.5],  
            ("unknown device class", "192.168.10.1", "2020-04-03 12:26:00"): [1., 1.2, 13.5],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:26:15"): [1.,  .7, 14.5],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:26:30"): [1., 1.0, 15.5],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:26:45"): [1., 1.1, 16.5],  
            ("unknown device class", "192.168.10.1", "2020-04-03 12:27:00"): [1., 1.2, 17.5] 
        }).T
        target.columns = ["traffic:bytes_rcvd", "ndpi_flows:num_flows__download-filetransfer-filesharing", "active_flows:flows_as_server"]
        target.index = pd.MultiIndex.from_tuples([(dc, h, pd.to_datetime(d, format="%Y-%m-%d %H:%M:%S")) for (dc, h, d) in target.index])
        target.index = target.index.rename(["device_category", "host", "_time"])

        pd.testing.assert_frame_equal(dset[target.columns], target, check_like=True)

    def test_multi_host_to_pandas(self):
        ii_host = mock_qresult.copy(deep=True)
        ii_host["host"] = "192.168.50.0"
        ii_host["_value"] += 15.
        mock_both_hosts = pd.concat([mock_qresult, ii_host]).reset_index()
        type(self.client.return_value).dframe = mock_both_hosts.copy(deep=True)
        _, sample = self.cicids2017.pull()
        dset = self.cicids2017.to_pandas(delta=False)

        target = pd.DataFrame({
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:15"): [0.,  40.,  101., 10.5],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:30"): [1.,  41.,  103., 11.5],  
            ("unknown device class", "192.168.10.1", "2020-04-03 12:25:45"): [2.,  42.1, 105., 12.5],
            ("unknown device class", "192.168.10.1", "2020-04-03 12:26:00"): [3.,  43.3, 53.,  13.5],
            ("unknown device class", "192.168.50.0", "2020-04-03 12:25:15"): [15., 55.,  131., 25.5],  
            ("unknown device class", "192.168.50.0", "2020-04-03 12:25:30"): [16., 56.,  133., 26.5],
            ("unknown device class", "192.168.50.0", "2020-04-03 12:25:45"): [17., 57.1, 135., 27.5],  
            ("unknown device class", "192.168.50.0", "2020-04-03 12:26:00"): [18., 58.3, 68.,  28.5],  
        }).T
        target.columns = [
            "traffic:bytes_rcvd", "ndpi_flows:num_flows__download-filetransfer-filesharing", 
            "ndpi_flows:num_flows__game", "active_flows:flows_as_server"]
        target.index = pd.MultiIndex.from_tuples([(dc, h, pd.to_datetime(d, format="%Y-%m-%d %H:%M:%S")) for (dc, h, d) in target.index])
        target.index = target.index.rename(["device_category", "host", "_time"])

        pd.testing.assert_frame_equal(dset[target.columns], target, check_like=True)

if __name__ == "__main__":
    unittest.main()
