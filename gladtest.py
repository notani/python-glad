#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import numpy as np
import unittest
import glad

verbose = False
debug = True
logger = None


class TestGLAD(unittest.TestCase):

    def setUp(self):
        self.data = glad.load_data('./data/data.txt')

    def test_EM(self):
        self.data.alpha = self.data.priorAlpha.copy()
        self.data.beta = self.data.priorBeta.copy()
        glad.EStep(self.data)
        np.testing.assert_allclose(self.data.probZ1[:5],
                                    [1.000000,1.000000,0.000000,0.000000,0.000000],
                                   atol=1e-3)
        np.testing.assert_allclose(self.data.probZ0[:5],
                                   [0.000000,0.000000,1.000000,1.000000,1.000000],
                                   atol=1e-3)
        Q = glad.computeQ(self.data)
        np.testing.assert_allclose(Q, -15490.194235, atol=1e-3)
        dQdAlpha, dQdBeta = glad.gradientQ(self.data)
        np.testing.assert_allclose(dQdAlpha[:5],
                                   [-623.990996,-621.437809,-526.274895,-556.247219,-553.387000],
                                   atol=1e-3)
        np.testing.assert_allclose(dQdBeta[:5],
                                   [-4.789440,-4.789440,-10.226004,-4.789440,-2.071159],
                                   atol=1e-3)

def init_logger():
    global logger
    logger = logging.getLogger('GLADTest')
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)

    glad.init_logger()


if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser()
    # parser.add_argument('filename')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    args = parser.parse_args()

    logger.info('Start Unit Test')
    unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGLAD)
    unittest.TextTestRunner(verbosity=2).run(suite)
    exit(0)
