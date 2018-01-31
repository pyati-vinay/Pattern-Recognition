# -*- coding: utf-8 -*-
"""
@author: vpp
"""

# imports
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as linalg
#import pandas as pd

# globals and utilities
label_dict = {1.0: 'Class 1', 2.0: 'Class 2', 3.0:'Class 3'}


def Normalize(X):
    # Normalize data
    mean = X.mean(axis=1).reshape(X.shape[0], 1)
    return X - np.tile(mean, (1, X.shape[1]))


def PCA(X, k):
    # Normalize data
    xNorm = Normalize(X)

    # Get covariance matrix of X
    cov_matrix = np.cov(xNorm)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors, = np.linalg.eigh(cov_matrix)

    # Considering absolute values
    eigenvalues = np.abs(eigenvalues)

    # sort eigenvalues by making pairs with eigenvectors
    eigen_pairs = [(eigenvalues[index], eigenvectors[:, index]) for index in range(len(eigenvalues))]
    eigen_pairs.sort()
    eigen_pairs.reverse()

    # Get eigenvalues and eigenvectors
    eigvectors_sort = [eigen_pairs[index][1] for index in range(len(eigenvalues))]

    # Considering the top 2/3 principal components
    top = np.array(eigvectors_sort[0:k]).transpose()

    # projection matrix
    projMatrix = np.dot(X.T, top)

    return projMatrix,eigenvalues

def LDA(X,y,k):
    dim = X.shape[0]
    mean = np.mean(X, axis=1).reshape(dim, 1)
    Sw = np.zeros((dim, dim))
    Sb = np.zeros((dim, dim))

    for label in label_dict:
        # fetch chunks of X for each class
        Xcls = X[:, y == label]
        nfeatures = Xcls.shape[1]
        # find mean of each class
        mCls = np.mean(Xcls, axis=1)
        mCls = mCls.reshape(dim, 1)
        covW = np.cov(Xcls - np.tile(mCls, (nfeatures,)),ddof=0)
        mDiff = (mCls - mean).reshape(dim, 1)
        # scatter matrix within and between
        Sw += covW
        Sb += mDiff.dot(mDiff.T)

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors, = np.linalg.eigh(linalg.inv(Sw).dot(Sb))

    # Considering absolute values
    eigenvalues = np.abs(eigenvalues)

    # sort eigenvalues by making pairs with eigenvectors
    eigen_pairs = [(eigenvalues[index], eigenvectors[:, index]) for index in range(len(eigenvalues))]
    eigen_pairs.sort()
    eigen_pairs.reverse()

    # Get eigenvalues and eigenvectors
    eigvectors_sort = [eigen_pairs[index][1] for index in range(len(eigenvalues))]

    # Considering the top 2/3 principal components
    top = np.array(eigvectors_sort[0:k]).transpose()

    # projection matrix
    projMatrix = np.dot(X.T, top)

    return projMatrix, eigenvalues


def ErrorAnalysis(eigenValues, k):
    # Error analysis
    keys = np.argsort(eigenValues)[::-1]
    eigenValues = eigenValues[keys]
    cummulativeSum = np.cumsum(eigenValues)
    errPerc = cummulativeSum[k - 1] / cummulativeSum[-1]
    print((1 - errPerc) * 100)


def plot_gen(X, y, k, isPCA=True):
    if k == 2:
        ax = plt.subplot(111)
        if isPCA :
            Y,eV = PCA(X, k)
        else:
            Y,eV = LDA(X, y, k)
        for label, color in zip(
                label_dict, ('C0', 'C1', 'C2')):
            plt.scatter(Y[:, 0][y == label],
                        Y[:, 1][y == label],
                        color=color,
                        alpha=0.5,
                        label=label_dict[label]
                        )
    else:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        if isPCA:
            Y,eV = PCA(X, k)
        else:
            Y,eV = LDA(X, y, k)

        for label, color in zip(
                label_dict, ('C0', 'C1', 'C2')):

            ax.scatter3D(Y[:, 0][y == label],
                         Y[:, 1][y == label],
                         Y[:, 2][y == label],
                         c=color,
                         label=label_dict[label])

        ax.set_zlabel("PC3" if isPCA else "LD3")

    ax.set_xlabel("PC1" if isPCA else "LD1")
    ax.set_ylabel("PC2" if isPCA else "LD2")
    leg = plt.legend(loc='upper left', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title("PCA "+str(k)+"D" if isPCA else "LDA "+str(k)+"D")

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.show()
    # Error analysis
    ErrorAnalysis(eV,k)
'''
# Reading dataset using pandas
datasetX = pd.read_csv("data-dimred-X.csv",header=None)
X = datasetX.iloc[:,0:150].values

# Get the outcome vector
datasetY = pd.read_csv("data-dimred-y.csv",header=None)
y = datasetY.iloc[:,0].values

'''
# Reading dataset using numpy
X = np.genfromtxt("data-dimred-X.csv", delimiter=',')
y = np.genfromtxt("data-dimred-y.csv", delimiter=',')


# PCA
plot_gen(X, y, 2)
plot_gen(X, y, 3)

# LDA
plot_gen(X, y, 2, isPCA=False)
plot_gen(X, y, 3, isPCA=False)