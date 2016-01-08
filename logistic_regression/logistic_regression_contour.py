# -*- coding: utf-8 -*-
"""
Created on Sun Nov 01 09:51:07 2015

@author: Timber
"""
import math
import functools
import pandas as pd
import numpy as np
from scipy.optimize import fmin_bfgs
from matplotlib import pyplot as plt

def logitProb(x, y, beta, intercept):
    returnMat = np.zeros(np.shape(x))
    for i in range(int(np.shape(x)[0])):
        for j in range(int(np.shape(x)[1])):
            xvec = baseExapnsion(np.asmatrix([x[i,j], y[i,j]]), 6)
            if (intercept):
                xvec = np.concatenate((np.asmatrix([1]), xvec), axis=1)
            returnMat[i,j] = sigmoid(np.inner(xvec, beta))
    return returnMat


def scatterPlot_simple(dfs):
    # subset the data frame according to admission result
    dfad = dfs[dfs.pa == 1][['Para1', 'Para2']]
    dfre = dfs[dfs.pa == 0][['Para1', 'Para2']]
    
    # scatter plot for the data points
    plt.scatter(dfad['Para1'], dfad['Para2'], marker='o', c='red') 
    plt.scatter(dfre['Para1'], dfre['Para2'], marker='^', c='blue')
    plt.xlabel("Para1")
    plt.ylabel("Para2")


def scatterPlot(dfs, beta, intercept):
    # subset the data frame according to admission result
    dfad = dfs[dfs.pa == 1][['Para1', 'Para2']]
    dfre = dfs[dfs.pa == 0][['Para1', 'Para2']]
    
    # scatter plot for the data points
    plt.scatter(dfad['Para1'], dfad['Para2'], marker='o', c='red') 
    plt.scatter(dfre['Para1'], dfre['Para2'], marker='^', c='blue')
    plt.xlabel("Para1")
    plt.ylabel("Para2")
    
    # plot the decision boundary (contour of logit probability)
    delta = 0.01
    x = np.arange(min(dfs['Para1']), max(dfs['Para1']), delta)
    y = np.arange(min(dfs['Para2']), max(dfs['Para2']), delta)
    X, Y = np.meshgrid(x, y)
    Z = logitProb(X, Y, beta, intercept)
    
    # use 0.5 probability as the decision threshold
    CS = plt.contour(X, Y, Z, [0.5])
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Scatter plot with logit decision boundary')

    
def sigmoid(x):
    return math.exp(x) / (1 + math.exp(x))


# given beta values, return the cost function value
def logisticCost(y, x, intercept, lam, beta):
    # input value y is an array, x is a matrix, beta is an array
    # when the user indicates intercept in regression
    # user has to make sure the dimension of 'beta' array is correct
    # (including intercept)
    if (intercept):
        ones = np.ones((len(y), 1))
        x = np.concatenate((ones, x), axis=1) # add a column of 1's
    
    # calculate cost function
    cost = 0
    for i in range(len(y)):
        # compute xbeta
        xbeta = 0;
        for j in range(len(beta)):
            xbeta += beta[j]*x[i, j]
        cost -= y[i]*xbeta
        cost += math.log(1+math.exp(xbeta))
    
    # add regulation term
    for j in range(1, len(beta)):
        cost += lam * (beta[j]**2)
    if (not(intercept)):
        cost += lam * (beta[0]**2)
    
    return cost

# given beta values, return the derivation value
def diff(y, x, intercept, lam, beta):
    # input value y is an array, x is a matrix, beta is an array
    diffVec = []
    
    # when the user indicates intercept in regression
    if (intercept):
        ones = np.ones((len(y), 1))
        x = np.concatenate((ones, x), axis=1) # add a column of 1's
    
    for betaIndex in range(len(beta)):
        diffVal = 0
        for i in range(len(y)):
            # compute xbeta
            xbeta = 0;
            for j in range(len(beta)):
                xbeta += beta[j]*x[i, j]
            diffVal -= y[i]*x[i, betaIndex]
            diffVal += x[i, betaIndex]*sigmoid(xbeta)
        # account for the regulation term
        if (betaIndex != 0 or not(intercept)):
            diffVal += (2 * lam * beta[betaIndex])
        diffVec.append(diffVal)
    
    return np.asarray(diffVec)
        
def baseExapnsion(xInput, deg):
    # expand the bases to the deg-th power
    # assume the input data is 2-D
    x = xInput
    for i in range(2, deg+1):
        for j in range(i+1):
            newCol = (np.asarray(x[:,0]) ** j) * (np.asarray(x[:,1]) ** (i-j))                    
            x = np.concatenate((x, newCol), axis=1)
    return x

def main():
    # read in the student admission data
    fp = open('ex2data2.txt', 'r')
    products = []
    for line in fp:
        row = line.strip().split(',')
        products.append([float(row[0]), float(row[1]), int(row[2])])
    
    dfs = pd.DataFrame(products)
    dfs.columns = ['Para1', 'Para2', 'pa']
    
    
    # create y array and x matrix
    ydata = np.array(dfs['pa'])
    xdata = np.asmatrix([np.array(dfs['Para1']), np.array(dfs['Para2'])])
    xdata = xdata.transpose()
    
    # perform base expansion
    xdata = baseExapnsion(xdata, 6)
    
    # regularization parameter
    lam = 0.1
    
    # use build in optimization function to calculate beta
    # beta initail guess
    betaInit = [0] * int(np.shape(xdata)[1])
    """
    betaOpt = fmin_bfgs(functools.partial(logisticCost, ydata, xdata, False, lam), betaInit, 
                        fprime = functools.partial(diff, ydata, xdata, False, lam))
    """
    betaInit.append(0)
    betaOptInt = fmin_bfgs(functools.partial(logisticCost, ydata, xdata, True, lam), betaInit, 
                        fprime = functools.partial(diff, ydata, xdata, True, lam))
    
    """
    scatterPlot(dfs, betaOpt, False)
    """
    scatterPlot(dfs, betaOptInt, True)

# when executed, just run main():
if __name__ == '__main__':
    main()


