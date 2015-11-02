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
import Gnuplot, Gnuplot.funcutils
import matplotlib
from matplotlib import pyplot as plt

def deciBoundary(x, beta, intercept):
    returnVec = []
    for i in range(len(x)):
        if (not(intercept)):
            returnVec.append((-1*beta[0]*x[i])/beta[1])
        else:
            returnVec.append((-1*beta[0] - beta[1]*x[i])/beta[2])
    return returnVec

def scatterPlot(dfs, beta, intercept):
    # subset the data frame according to admission result
    dfad = dfs[dfs.ad == 1][['Exam1', 'Exam2']]
    dfre = dfs[dfs.ad == 0][['Exam1', 'Exam2']]
    
    plt.scatter(dfad['Exam1'], dfad['Exam2'], marker='o', c='red') 
    plt.scatter(dfre['Exam1'], dfre['Exam2'], marker='^', c='blue')
    
    plotLim = plt.axis()
    plt.plot([plt.axis()[0], plt.axis()[1]], deciBoundary([plt.axis()[0], plt.axis()[1]], beta, intercept))
    plt.axis(plotLim)
    
    # Plot scatter plot using different symbols to represent
    # admitted students and rejected students
    #gplot = Gnuplot.Gnuplot(debug=1)
    #gplot.title("Student exam scatter plot")
    #gplot('set data style linespoints')
    #plotData = []
    #plotData.append(Gnuplot.Data(dfre))
    #plotData.append(Gnuplot.Data(dfad))
    
    # function of decision boundary
    #gplot.plot(plotData[0], plotData[1])
    
    #gplot.plot(plotData[0], plotData[1])
    
    #raw_input('Please press return to continue...\n')
    #gplot.reset()
    
def sigmoid(x):
    return math.exp(x) / (1 + math.exp(x))

def diffLike(df, beta, index):
    # users have to make sure the dimensions of input vecotrs/matrices
    # match each other, the function won't do dimension inspection
    diffl = 0
    for i in range(len(df)):
        xbeta = 0
        for j in range(len(df.columns) - 1):
            xbeta += df.loc[i][ df.columns[j] ]*beta.loc[0][ df.columns[j] ]
        
        diffl += df.loc[i][index] * df.loc[i]['ad']
        diffl -= df.loc[i][index] * sigmoid(xbeta)
    
    return diffl

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


def gradDecent(df):
    # converges very slowly...need improvement
    alpha = 0.0005
    epsilon = 10 ** -5
    
    # apply gradient decent to one beta at a time
    oldbeta1 = 180.0
    oldbeta2 = 130.0
    dfbeta = pd.DataFrame({'Exam1':[oldbeta1], 'Exam2':[oldbeta2]})
    newbeta1 = oldbeta1 + alpha*diffLike(df, dfbeta, 'Exam1')
    newbeta2 = oldbeta2 + alpha*diffLike(df, dfbeta, 'Exam2')
    dfbeta.loc[0]['Exam1'] = newbeta1
    dfbeta.loc[0]['Exam2'] = newbeta2
    while abs(newbeta1 - oldbeta1)>epsilon or abs(newbeta2 - oldbeta2)>epsilon:
        oldbeta1 = newbeta1
        oldbeta2 = newbeta2
        newbeta1 = oldbeta1 + alpha*diffLike(df, dfbeta, 'Exam1')
        newbeta2 = oldbeta2 + alpha*diffLike(df, dfbeta, 'Exam2')
        dfbeta.loc[0]['Exam1'] = newbeta1
        dfbeta.loc[0]['Exam2'] = newbeta2
        print dfbeta
    
    return dfbeta
        

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

scatterPlot(dfs, betaOpt, False)
scatterPlot(dfs, betaOptInt, True)







