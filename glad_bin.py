#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import math
import numpy as np
import scipy as sp
import scipy.stats
import scipy.optimize
import unittest
import warnings


THRESHOLD = 1e-5

verbose = False
debug = False
logger = None
# warnings.filterwarnings('error')


class Dataset(object):
    def __init__(self, labels=None, numLabels=-1, numLabelers=-1, numTasks=-1,
                 priorAlpha=None, priorBeta=None, priorZ1=None,
                 alpha=None, beta=None, probZ1=None, probZ0=None):
        self.labels = labels
	self.numLabels = numLabels
	self.numLabelers = numLabelers
	self.numTasks = numTasks
	self.priorAlpha = priorAlpha
	self.priorBeta = priorBeta
	self.priorZ1 = priorZ1
	self.alpha = alpha
        self.beta = beta
	self.probZ1 = probZ1
        self.probZ0 = probZ0

def init_logger():
    global logger
    logger = logging.getLogger('GLAD')
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logsigmoid(x):
    return - np.log(1 + np.exp(-x))

def load_data(filename):
    data = Dataset()
    with open(filename) as f:
        # Read parameters
        header = f.readline().split()
        data.numLabels = int(header[0])
        data.numLabelers = int(header[1])
        data.numTasks = int(header[2])
        data.priorZ1 = float(header[3])
        if verbose:
            logger.info('Reading {} labels of {} labelers over {} tasks for prior P(Z=1) = {}'.format(data.numLabels, data.numLabelers, data.numTasks, data.priorZ1))
        # Read Labels
        data.labels = np.zeros((data.numTasks, data.numLabelers))
        for line in f:
            task, labeler, label = map(int, line.split())
            if verbose:
                logger.info("Read: task({})={} by labeler {}".format(task, label, labeler))
            data.labels[task][labeler] = label + 1
    # Initialize Probs
    data.priorAlpha = np.ones(data.numLabelers)
    data.priorBeta = np.ones(data.numTasks)
    data.probZ1 = np.empty(data.numTasks)
    data.probZ0 = np.empty(data.numTasks)
    data.priorZ1 = np.zeros(data.numTasks) + data.priorZ1
    data.beta = np.empty(data.numTasks)
    data.alpha = np.empty(data.numLabelers)

    return data

def EM(data):
    u"""Infer true labels, tasks' difficulty and workers' ability
    """
    # Initialize parameters to starting values
    data.alpha = data.priorAlpha.copy()
    data.beta = data.priorBeta.copy()

    EStep(data)
    lastQ = computeQ(data)
    dQdAlpha, dQdBeta = gradientQ(data)
    MStep(data)
    Q = computeQ(data)
    counter = 1
    while abs((Q - lastQ)/lastQ) > THRESHOLD:
        if verbose: logger.info('EM: iter={}'.format(counter))
        lastQ = Q
        EStep(data)
        MStep(data)
        Q = computeQ(data)
        counter += 1

def EStep(data):
    u"""Evaluate the posterior probability of true labels given observed labels and parameters
    """
    def calcLogProbL(item, *args):
        i = item[0]
        idx = args[0][int(i)]
        row = item[1:]
        return logsigmoid(row[idx]).sum() + logsigmoid(-row[np.invert(idx)]).sum()

    if verbose: logger.info('EStep')
    data.probZ1 = np.log(data.priorZ1)
    data.probZ0 = np.log(1 - data.priorZ1)

    ab = np.dot(np.array([np.exp(data.beta)]).T, np.array([data.alpha]))
    ab[data.labels == 0] = 0  # drop ab with no response
    ab = np.c_[np.arange(data.numTasks), ab]

    # TODO: Z1 -> Z2
    data.probZ1 = np.apply_along_axis(calcLogProbL, 1, ab, (data.labels == 2))
    data.probZ0 = np.apply_along_axis(calcLogProbL, 1, ab, (data.labels == 1))

    # Exponentiate and renormalize
    data.probZ1 = np.exp(data.probZ1)
    data.probZ0 = np.exp(data.probZ0)
    data.probZ1 = data.probZ1 / (data.probZ1 + data.probZ0)
    data.probZ0 = 1 - data.probZ1
    # TODO: nan -> abort

    return data

def packX(data):
    return np.r_[data.alpha.copy(), data.beta.copy()]

def unpackX(x, data):
    data.alpha = x[:data.numLabelers].copy()
    data.beta = x[data.numLabelers:].copy()

def getBoundsX(data, alpha=(-100, 100), beta=(-100, 100)):
    alpha_bounds = np.array([[alpha[0], alpha[1]] for i in range(data.numLabelers)])
    beta_bounds = np.array([[beta[0], beta[1]] for i in range(data.numLabelers)])
    return np.r_[alpha_bounds, beta_bounds]

def f(x, *args):
    u"""Return the value of the objective function
    """
    data = args[0]
    d = Dataset(labels=data.labels, numLabels=data.numLabels, numLabelers=data.numLabelers,
                numTasks=data.numTasks,
                priorAlpha=data.priorAlpha, priorBeta=data.priorBeta, priorZ1=data.priorZ1,
                probZ1=data.probZ1, probZ0=data.probZ0)
    unpackX(x, d)
    return - computeQ(d)

def df(x, *args):
    u"""Return gradient vector
    """
    data = args[0]
    d = Dataset(labels=data.labels, numLabels=data.numLabels, numLabelers=data.numLabelers,
                numTasks=data.numTasks,
                priorAlpha=data.priorAlpha, priorBeta=data.priorBeta, priorZ1=data.priorZ1,
                probZ1=data.probZ1, probZ0=data.probZ0)
    unpackX(x, d)
    dQdAlpha, dQdBeta = gradientQ(d)
    # Flip the sign since we want to minimize
    return np.r_[-dQdAlpha, -dQdBeta]

def MStep(data):
    if verbose: logger.info('MStep')
    initial_params = packX(data)
    params = sp.optimize.minimize(fun=f, x0=initial_params, args=(data,), method='CG',
                                  jac=df, tol=0.01,
                                  options={'maxiter': 25, 'disp': verbose})
    if debug: logger.debug(params)
    unpackX(params.x, data)

def computeQ(data):
    u"""Calculate the expectation of the joint likelihood
    """
    Q = 0
    # Start with the expectation of the sum of priors over all tasks
    Q += (data.probZ1 * np.log(data.priorZ1)).sum()
    Q += (data.probZ0 * np.log(1 - data.priorZ1)).sum()

    # the expectation of the sum of posteriors over all tasks
    ab = np.dot(np.array([np.exp(data.beta)]).T, np.array([data.alpha]))

    logSigma = - np.log(1 + np.exp(-ab))
    idxna = np.isnan(logSigma)
    if np.any(idxna): logger.warning('an invalid value was assigned to np.log [computeQ]')
    logSigma[idxna] = ab[idxna]  # For large negative x, -log(1 + exp(-x)) = x

    logOneMinusSigma = - np.log(1 + np.exp(ab))
    idxna = np.isnan(logOneMinusSigma)
    if np.any(idxna): logger.warning('an invalid value was assigned to np.log [computeQ]')
    logOneMinusSigma[idxna] = -ab[idxna]  # For large positive x, -log(1 + exp(x)) = x

    # class = 1 :TODO
    idx = (data.labels == 2)
    Q += (data.probZ1 * logSigma.T).T[idx].sum()
    Q += (data.probZ1 * logOneMinusSigma.T).T[np.invert(idx)].sum()
    # class = 1 :TODO
    idx = (data.labels == 1)
    Q += (data.probZ0 * logSigma.T).T[idx].sum()
    Q += (data.probZ0 * logOneMinusSigma.T).T[np.invert(idx)].sum()

    # Add Gaussian (standard normal) prior for alpha
    Q += np.log(sp.stats.norm.pdf(data.alpha - data.priorAlpha)).sum()

    # Add Gaussian (standard normal) prior for beta
    Q += np.log(sp.stats.norm.pdf(data.beta - data.priorBeta)).sum()

    if debug:
        logger.debug('a[0]={} a[1]={} a[2]={} b[0]={}'.format(data.alpha[0], data.alpha[1],
                                                              data.alpha[2], data.beta[0]))
        logger.debug('Q={}'.format(Q))
    if np.isnan(Q):
        return -np.inf
    return Q


def gradientQ(data):
    def dAlpha(item, *args):
        idx = args[0][:, int(item[0])]
        probZ = args[1]
        row = item[1:]
        correct = ((1 - row) * probZ)[idx]
        wrong = -(row * probZ)[np.invert(idx)]
        return correct.sum() + wrong.sum()

    def dBeta(item, *args):
        idx = args[0][int(item[0])]
        alpha = args[1]
        row = item[1:]
        correct = ((1 - row) * alpha)[idx]
        wrong = -(row * alpha)[np.invert(idx)]
        return correct.sum() + wrong.sum()

    dQdAlpha = - (data.alpha - data.priorAlpha)
    dQdBeta = - (data.beta - data.priorBeta)

    ab = np.dot(np.array([np.exp(data.beta)]).T, np.array([data.alpha]))

    sigma = sigmoid(ab)
    sigma[data.labels == 0] = 0  # drop ab with no response
    sigma[np.isnan(sigma)] = 0  # :TODO check if this is correct

    labelersIdx = np.arange(data.numLabelers).reshape((1,data.numLabelers))
    sigma = np.r_[labelersIdx, sigma]
    sigma = np.c_[np.arange(-1, data.numTasks), sigma]


    # print(data.probZ1 * np.exp(data.beta))
    dQdAlpha += np.apply_along_axis(dAlpha, 0, sigma[:, 1:],
                                    (data.labels == 2), data.probZ1 * np.exp(data.beta))
    dQdAlpha += np.apply_along_axis(dAlpha, 0, sigma[:, 1:],
                                    (data.labels == 1), data.probZ0 * np.exp(data.beta))

    dQdBeta += np.apply_along_axis(dBeta, 1, sigma[1:],
                                   (data.labels == 2), data.alpha) * data.probZ1 * np.exp(data.beta)
    dQdBeta += np.apply_along_axis(dBeta, 1, sigma[1:],
                                   (data.labels == 1), data.alpha) * data.probZ0 * np.exp(data.beta)

    if debug:
        logger.debug('dQdAlpha[0]={} dQdAlpha[1]={} dQdAlpha[2]={} dQdBeta[0]={}'.format(dQdAlpha[0], dQdAlpha[1],
                                                                                         dQdAlpha[2], dQdBeta[0]))
    return dQdAlpha, dQdBeta


def output(data):
    alpha = np.c_[np.arange(1, data.numLabelers+1), data.alpha]
    np.savetxt('alpha.csv', alpha, fmt=['%d', '%.5f'], delimiter=',', header='id,alpha')
    beta = np.c_[np.arange(1, data.numTasks+1), np.exp(data.beta)]
    np.savetxt('beta.csv', beta, fmt=['%d', '%.5f'], delimiter=',', header='id,beta')
    label = np.c_[np.arange(1, data.numTasks+1), data.probZ0, data.probZ1]
    np.savetxt('label.csv', label, fmt=['%d', '%.5f', '%.5f'], delimiter=',', header='id,z0,z1')


def outputResults(data):
    for i in range(data.numLabelers):
        print('Alpha[{idx}] = {val:.5f}'.format(idx=i, val=data.alpha[i]))

    for j in range(data.numTasks):
        print('Beta[{idx}] = {val:.5f}'.format(idx=j, val=np.exp(data.beta[j])))

    for j in range(data.numTasks):
        print('P(Z({idx})=1) = {val:.5f}'.format(idx=j, val=data.probZ1[j]))


def main(args=None):
    global debug, verbose
    debug = args.debug
    if debug == True:
        verbose = True
    else:
        verbose = args.verbose

    data = load_data(args.filename)
    EM(data)

    output(data)
    # outputResults(data)
    return


if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    args = parser.parse_args()

    code = main(args)
    exit(code)
