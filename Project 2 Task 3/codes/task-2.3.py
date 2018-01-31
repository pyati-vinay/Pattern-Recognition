# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 14:19:48 2017

@author: vpp
"""

# utilities
import numpy as np
import matplotlib.pyplot as plt

# load file
data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)

# read height and weight
xData = data[:, 0:2].astype(np.float)

# Transpose
xData = xData.T

# removing the outliers
m = xData[:, xData.min(axis=0) >= 0]

# copy
weights = np.copy(m[0, :])
heights = np.copy(m[1, :])

# Define outcome vector
y = weights

# Define co-variate
x = heights

# Array filled with ones
ones = np.ones(len(y))

# High degree co-variates
quadratic = np.power(x, 2)
cubic = np.power(x, 3)
quartic = np.power(x, 4)
quintic = np.power(x, 5)

# Build design matrix
X = np.column_stack((ones, x, quadratic, cubic, quartic, quintic))

# Assuming Gaussian prior for parameter vector w
bayesComponent = 3*np.identity(len(X[1, :]))

# Estimation
xTranspose = np.matrix.transpose(X)
w = np.dot(np.linalg.inv(np.add(np.dot(xTranspose, X), bayesComponent)), np.dot(xTranspose, y))

# Plot the body height and weight points
plt.scatter(heights, weights)

# Title and labels
plt.title('Body Height-Weight')
plt.ylabel('Weight')
plt.xlabel('Height')

x_new = np.sort(x)
y_new = w[0] + np.sort(x)*w[1] + np.sort(quadratic)*w[2] + np.sort(cubic)*w[3]\
        + np.sort(quartic)*w[4] + np.sort(quintic)*w[5]


# Plot bayesian regression
bayesian, = plt.plot(x_new, y_new, 'r-')

# Legend
plt.legend([bayesian], ['Bayesian 5th degree polynomial'], loc='upper left')

plt.show()
