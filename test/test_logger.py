import numpy as np
import unittest
from tesi_sabella.logger import ResultLogger


# ----- ----- LOGGER ----- ----- #
# ----- ----- ------ ----- ----- #
class TestLogger(unittest.TestCase):
    def setUp(self, mock_influx_client):
        self.logger = ResultLogger('test_logger_tmp/')
        
        # Storing values in logger
        c = ['tr_lss', 'ts_lss', 'tr_prec', 'ts_prec']
        cm = np.random.rand((4, 50))
        self.cm_dict = {c[i]: cm[i,:] for i in range(len(c))}
        
        for i in cm.shape[1]:
            batch = {k: self.cm_dict[k][i] for k in c}
            self.logger.log(batch)

    def test_log(self):
        self.assertEqual(self.cm_dict, self.logger.log_tree, 'incorrect metrics log')

    def test_log_store(self):
        self.logger.dump('fake_training')
        restored_logger = ResultLogger('test_logger_tmp/')
        restored_logger.load('fake_training')
        self.assertEqual(self.cm_dict, restored_logger.log_tree, 'Stored file mismatch')
