import numpy as np
import unittest
from tesi_sabella.logger import ResultLogger


# ----- ----- LOGGER ----- ----- #
# ----- ----- ------ ----- ----- #
class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = ResultLogger('test_logger_tmp/')
        

    def test_log(self):
        # Storing values in logger
        c = ['tr_lss', 'ts_lss', 'tr_prec', 'ts_prec']
        cm = np.random.rand(4, 50)
        cm_dict = {c[i]: cm[i,:] for i in range(len(c))}
        
        for i in range(cm.shape[1]):
            batch = {k: cm_dict[k][i] for k in c}
            self.logger.log(batch)

        self.assertCountEqual(cm_dict, self.logger.tmp_batch, 'incorrect metrics log')

    def test_log_store(self):
        self.logger.log({'tr_lss': 1})
        self.logger.log({'tr_lss': 2})
        self.logger.log({'tr_lss': 3})
        self.logger.collapse('tr_lss')

        self.logger.dump('fake_training')
        restored_logger = ResultLogger('test_logger_tmp/')
        restored_logger.load('fake_training.log')
        self.assertEqual(np.array([2]), restored_logger.log_tree['tr_lss'], 'Stored file mismatch')
