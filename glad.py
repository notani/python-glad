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
    def __init__(self, labels=None,
                 numLabels=-1, numLabelers=-1, numTasks=-1, numClasses=-1,
                 priorAlpha=None, priorBeta=None, priorZ=None,
                 alpha=None, beta=None, probZ=None):
        self.labels = labels
        self.numLabels = numLabels
        self.numLabelers = numLabelers
        self.numTasks = numTasks
        self.numClasses = numClasses
        self.priorAlpha = priorAlpha
        self.priorBeta = priorBeta
        self.priorZ = priorZ
        self.alpha = alpha
        self.beta = beta
        self.probZ = probZ


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
        data.numClasses = int(header[3])
        data.priorZ = np.array([float(x) for x in header[4:]])
        assert len(data.priorZ) == data.numClasses, 'Incorrect input header'
        assert data.priorZ.sum() == 1, 'Incorrect priorZ given'
        if verbose:
            logger.info('Reading {} labels of {} labelers over {} tasks for prior P(Z) = {}'.format(data.numLabels,
                                                                                                    data.numLabelers,
                                                                                                    data.numTasks,
                                                                                                    data.priorZ))
        # Read Labels
        data.labels = np.zeros((data.numTasks, data.numLabelers))
        for line in f:
            task, labeler, label = map(int, line.split())
            if debug:
                logger.info("Read: task({})={} by labeler {}".format(task, label, labeler))
            data.labels[task][labeler] = label + 1
    # Initialize Probs
    data.priorAlpha = np.ones(data.numLabelers)
    data.priorBeta = np.ones(data.numTasks)
    data.probZ = np.empty((data.numTasks, data.numClasses))
    # data.priorZ = (np.zeros((data.numClasses, data.numTasks)).T + data.priorZ).T
    data.beta = np.empty(data.numTasks)
    data.alpha = np.empty(data.numLabelers)

    return data


def EM(data):
    u"""Infer true labels, tasks' difficulty and workers' ability
    """
    # Initialize parameters to starting values
    data.alpha = data.priorAlpha.copy()
    data.beta = data.priorBeta.copy()
    data.probZ[:] = data.priorZ[:]

    EStep(data)
    lastQ = computeQ(data)
    MStep(data)
    Q = computeQ(data)
    counter = 1
    while abs((Q - lastQ) / lastQ) > THRESHOLD:
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
        j = int(item[0])  # task ID

        # List[boolean]: denotes if the worker i picked the focused class for the task j
        ## formally, delta[i, j] = True if l_ij == z_j for i = 0, ..., m-1 (m=# of workers)
        delta = args[0][j]
        noResp = args[1][j]
        oneMinusDelta = (~delta) & (~noResp)

        # List[float]: alpha_i * exp(beta_j) for i = 0, ..., m-1
        exponents = item[1:]

        # Log likelihood for the observations s.t. l_ij == z_j
        correct = logsigmoid(exponents[delta]).sum()
        # Log likelihood for the observations s.t. l_ij != z_j
        wrong = (logsigmoid(-exponents[oneMinusDelta]) - np.log(float(data.numClasses - 1))).sum()

        # Return log likelihood
        return correct + wrong

    if verbose: logger.info('EStep')
    data.probZ = np.tile(np.log(data.priorZ), data.numTasks).reshape(data.numTasks, data.numClasses)

    ab = np.dot(np.array([np.exp(data.beta)]).T, np.array([data.alpha]))
    ab = np.c_[np.arange(data.numTasks), ab]

    for k in range(data.numClasses):
        data.probZ[:, k] = np.apply_along_axis(calcLogProbL, 1, ab,
                                               (data.labels == k + 1),
                                               (data.labels == 0))

    # Exponentiate and renormalize
    data.probZ = np.exp(data.probZ)
    s = data.probZ.sum(axis=1)
    data.probZ = (data.probZ.T / s).T
    assert not np.any(np.isnan(data.probZ)), 'Invalid Value [EStep]'
    assert not np.any(np.isinf(data.probZ)), 'Invalid Value [EStep]'

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
                numTasks=data.numTasks, numClasses=data.numClasses,
                priorAlpha=data.priorAlpha, priorBeta=data.priorBeta,
                priorZ=data.priorZ, probZ=data.probZ)
    unpackX(x, d)
    return - computeQ(d)


def df(x, *args):
    u"""Return gradient vector
    """
    data = args[0]
    d = Dataset(labels=data.labels, numLabels=data.numLabels, numLabelers=data.numLabelers,
                numTasks=data.numTasks, numClasses=data.numClasses,
                priorAlpha=data.priorAlpha, priorBeta=data.priorBeta,
                priorZ=data.priorZ, probZ=data.probZ)
    unpackX(x, d)
    dQdAlpha, dQdBeta = gradientQ(d)
    # Flip the sign since we want to minimize
    assert not np.any(np.isinf(dQdAlpha)), 'Invalid Gradient Value [Alpha]'
    assert not np.any(np.isinf(dQdBeta)), 'Invalid Gradient Value [Beta]'
    assert not np.any(np.isnan(dQdAlpha)), 'Invalid Gradient Value [Alpha]'
    assert not np.any(np.isnan(dQdBeta)), 'Invalid Gradient Value [Beta]'
    return np.r_[-dQdAlpha, -dQdBeta]


def MStep(data):
    if verbose: logger.info('MStep')
    initial_params = packX(data)
    params = sp.optimize.minimize(fun=f, x0=initial_params, args=(data,), method='CG',
                                  jac=df, tol=0.01,
                                  options={'maxiter': 25, 'disp': verbose})
    if debug:
        logger.debug(params)
    unpackX(params.x, data)


def computeQ(data):
    u"""Calculate the expectation of the joint likelihood
    """
    Q = 0
    # Start with the expectation of the sum of priors over all tasks
    Q += (data.probZ * np.log(data.priorZ)).sum()

    # the expectation of the sum of posteriors over all tasks
    ab = np.dot(np.array([np.exp(data.beta)]).T, np.array([data.alpha]))

    # logSigma = - np.log(1 + np.exp(-ab))
    logSigma = logsigmoid(ab)  # logP
    idxna = np.isnan(logSigma)
    if np.any(idxna):
        logger.warning('an invalid value was assigned to np.log [computeQ]')
        logSigma[idxna] = ab[idxna]  # For large negative x, -log(1 + exp(-x)) = x

    # logOneMinusSigma = - np.log(1 + np.exp(ab))
    logOneMinusSigma = logsigmoid(-ab) - np.log(float(data.numClasses - 1))  # log((1-P)/(K-1))
    idxna = np.isnan(logOneMinusSigma)
    if np.any(idxna):
        logger.warning('an invalid value was assigned to np.log [computeQ]')
        logOneMinusSigma[idxna] = -ab[idxna]  # For large positive x, -log(1 + exp(x)) = x

    for k in range(data.numClasses):
        delta = (data.labels == k + 1)
        Q += (data.probZ[:, k] * logSigma.T).T[delta].sum()
        oneMinusDelta = (data.labels != k + 1) & (data.labels != 0)  # label == 0 -> no response
        Q += (data.probZ[:, k] * logOneMinusSigma.T).T[oneMinusDelta].sum()

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
        i = int(item[0])  # worker ID
        sigma_ab = item[1:] # List[float], dim=(n,): sigmoid(alpha_i * beta_j) for j = 0, ..., n-1

        # List[boolean], dim=(n,): denotes if the worker i picked the focused class for
        # task j (j=0, ..., n-1)
        delta = args[0][:, i]
        noResp = args[1][:, i]
        oneMinusDelta = (~delta) & (~noResp)

        # List[float], dim=(n,): Prob of the true label of the task j being the focused class (p^k)
        probZ = args[2]

        correct = probZ[delta] * np.exp(data.beta[delta]) * (1 - sigma_ab[delta])
        wrong = probZ[oneMinusDelta] * np.exp(data.beta[oneMinusDelta]) * (-sigma_ab[oneMinusDelta])
        # Note: The derivative in Whitehill et al.'s appendix has the term ln(K-1), which is incorrect.

        return correct.sum() + wrong.sum()

    def dBeta(item, *args):
        j = int(item[0])  # task ID
        sigma_ab = item[1:] # List[float], dim=(m,): sigmoid(alpha_i * beta_j) for i = 0, ..., m-1

        # List[boolean], dim=(m,): denotes if the worker i picked the focused class for
        # task j (i=0, ..., m-1)
        delta = args[0][j]
        noResp = args[1][j]
        oneMinusDelta = (~delta) & (~noResp)

        # float: Prob of the true label of the task j being the focused class (p^k)
        probZ = args[2][j]

        correct = probZ * data.alpha[delta] * (1 - sigma_ab[delta])
        wrong = probZ * data.alpha[oneMinusDelta] * (-sigma_ab[oneMinusDelta])

        return correct.sum() + wrong.sum()

    # prior prob.
    dQdAlpha = - (data.alpha - data.priorAlpha)
    dQdBeta = - (data.beta - data.priorBeta)

    ab = np.dot(np.array([np.exp(data.beta)]).T, np.array([data.alpha]))

    sigma = sigmoid(ab)
    sigma[np.isnan(sigma)] = 0  # :TODO check if this is correct

    labelersIdx = np.arange(data.numLabelers).reshape((1, data.numLabelers))
    sigma = np.r_[labelersIdx, sigma]
    sigma = np.c_[np.arange(-1, data.numTasks), sigma]
    # sigma: List[List[float]]: dim=(n+1, m+1) where n = # of tasks and m = # of workers
    # sigma[0] = List[float]: worker IDs (-1, 0, ..., m-1) where the first -1 is a pad
    # sigma[:, 0] = List[float]: task IDs (-1, 0, ..., n-1) where the first -1 is a pad

    for k in range(data.numClasses):
        dQdAlpha += np.apply_along_axis(dAlpha, 0, sigma[:, 1:],
                                        (data.labels == k + 1),
                                        (data.labels == 0),
                                        data.probZ[:, k])

        dQdBeta += np.apply_along_axis(dBeta, 1, sigma[1:],
                                       (data.labels == k + 1),
                                        (data.labels == 0),
                                       data.probZ[:, k]) * np.exp(data.beta)

    if debug:
        logger.debug('dQdAlpha[0]={} dQdAlpha[1]={} dQdAlpha[2]={} dQdBeta[0]={}'.format(dQdAlpha[0], dQdAlpha[1],
                                                                                         dQdAlpha[2], dQdBeta[0]))
    return dQdAlpha, dQdBeta


def output(data):
    alpha = np.c_[np.arange(data.numLabelers), data.alpha]
    np.savetxt('data/alpha.csv', alpha, fmt=['%d', '%.5f'], delimiter=',', header='id,alpha')
    beta = np.c_[np.arange(data.numTasks), np.exp(data.beta)]
    np.savetxt('data/beta.csv', beta, fmt=['%d', '%.5f'], delimiter=',', header='id,beta')
    probZ = np.c_[np.arange(data.numTasks), data.probZ]
    np.savetxt(fname='data/probZ.csv',
               X=probZ,
               fmt=['%d'] + (['%.5f'] * data.numClasses),
               delimiter=',',
               header='id,' + ','.join(['z' + str(k) for k in range(data.numClasses)]))
    label = np.c_[np.arange(data.numTasks), np.argmax(data.probZ, axis=1)]
    np.savetxt('data/label_glad.csv', label, fmt=['%d', '%d'], delimiter=',', header='id,label')


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
