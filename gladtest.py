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

    def test_computeQ(self):
        Q = glad.computeQ(self.data)
        np.testing.assert_allclose(Q, -1447.31730387, atol=1e-3)

    def test_EM(self):
        self.data.alpha = self.data.priorAlpha.copy()
        self.data.beta = self.data.priorBeta.copy()
        glad.EStep(self.data)
        np.testing.assert_allclose(self.data.probZ1[:3],
                                   [1.00000000e+00,1.00000000e+00,1.56549620e-12],
                                   atol=1e-3)
        np.testing.assert_allclose(self.data.probZ0[:3],
                                   [0,0,1],
                                   atol=1e-3)
        Q = glad.computeQ(self.data)
        np.testing.assert_allclose(Q, -15490.1942347, atol=1e-3)
        dQdAlpha, dQdBeta = glad.gradientQ(self.data)
        np.testing.assert_allclose(dQdAlpha[:3],
                                   [-623.99099612, -621.43780919, -526.27489508],
                                   atol=1e-3)
        np.testing.assert_allclose(dQdBeta[:3],
                                   [-4.78944038, -4.78944038, -10.22600404],
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
