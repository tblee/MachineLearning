# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:02:41 2015

@author: Timber
"""

import math
import functools
import pandas as pd
import numpy as np
from scipy.optimize import fmin_bfgs
from matplotlib import pyplot as plt

def deciBoundary(x, beta, intercept):
    returnVec = []
    for i in range(len(x)):
        if (not(intercept)):
            returnVec.append((-1*beta[0]*x[i])/beta[1])
        else:
            returnVec.append((-1*beta[0] - beta[1]*x[i])/beta[2])
    return returnVec

def scatterPlot(dfs, beta, intercept, boundLabel):
    # subset the data frame according to admission result
    dfad = dfs[dfs.ad == 1][['Exam1', 'Exam2']]
    dfre = dfs[dfs.ad == 0][['Exam1', 'Exam2']]
    
    # scatter plot for the data points
    plt.scatter(dfad['Exam1'], dfad['Exam2'], marker='o', c='red') 
    plt.scatter(dfre['Exam1'], dfre['Exam2'], marker='^', c='blue')
    plt.xlabel("Exam1")
    plt.ylabel("Exam2")
    
    # add decision boundary line onto the plot
    plotLim = plt.axis()
    plt.plot([plt.axis()[0], plt.axis()[1]], deciBoundary([plt.axis()[0], plt.axis()[1]], 
                       beta, intercept), label=boundLabel)
    plt.axis(plotLim)
    plt.legend()

    
def sigmoid(x):
    return math.exp(x) / (1 + math.exp(x))


# given beta values, return the cost function value
def logisticCost(y, x, intercept, beta):
    # input value y is an array, x is a matrix, beta is an array
    # when the user indicates intercept in regression
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
    return cost

# given beta values, return the derivation value
def diff(y, x, intercept, beta):
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
        diffVec.append(diffVal)
    return np.asarray(diffVec)
        
def main():
    # read in the student admission data
    fp = open('ex2data1.txt', 'r')
    students = []
    for line in fp:
        row = line.strip().split(',')
        students.append([float(row[0]), float(row[1]), int(row[2])])
    
    dfs = pd.DataFrame(students)
    dfs.columns = ['Exam1', 'Exam2', 'ad']
    # normalize raw data to prevent math overflow
    dfs['Exam1'] = (dfs['Exam1'] - np.mean(dfs['Exam1'])) / np.std(dfs['Exam1'])
    dfs['Exam2'] = (dfs['Exam2'] - np.mean(dfs['Exam2'])) / np.std(dfs['Exam2'])
    
    # create y array and x matrix
    ydata = np.array(dfs['ad'])
    xdata = np.asmatrix([np.array(dfs['Exam1']), np.array(dfs['Exam2'])])
    xdata = xdata.transpose()
    
    # use build in optimization function to calculate beta
    betaOpt = fmin_bfgs(functools.partial(logisticCost, ydata, xdata, False), [0, 0], 
                        fprime = functools.partial(diff, ydata, xdata, False))
    betaOptInt = fmin_bfgs(functools.partial(logisticCost, ydata, xdata, True), [0, 0, 0], 
                        fprime = functools.partial(diff, ydata, xdata, True))
    
    scatterPlot(dfs, betaOpt, False, "w/o intercept")
    scatterPlot(dfs, betaOptInt, True, "w/ intercept")

# when executed, just run main():
if __name__ == '__main__':
    main()





