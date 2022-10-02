# Author: Kristjan Orri Dadason
# Date: 28/09/2022
# Project: 07_linear_models
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

from json import load
import numpy as np
import matplotlib.pyplot as plt

from tools import load_regression_iris
from scipy.stats import multivariate_normal


def mvn_basis(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float
) -> np.ndarray:
    '''
    Multivariate Normal Basis Function
    The function transforms (possibly many) data vectors <features>
    to an output basis function output <fi>
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional
    data vectors.
    * mu: [MxD] matrix of M D-dimensional mean vectors defining
    the multivariate normal distributions.
    * sigma: All normal distributions are isotropic with sigma*I covariance
    matrices (where I is the MxM identity matrix)
    Output:
    * fi - [NxM] is the basis function vectors containing a basis function
    output fi for each data vector x in features
    '''
    return np.array([multivariate_normal(mean=mu[i], cov=sigma*np.eye(mu.shape[1])).pdf(features) for i in range(mu.shape[0])]).T

def _plot_mvn():
    X, t = load_regression_iris()
    N, D = X.shape
    #print(X[:5, :])
    M, sigma = 10, 10
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
    fi = mvn_basis(X, mu, sigma)
    plt.plot(fi)
    plt.show()
    

def max_likelihood_linreg(
    fi: np.ndarray,
    targets: np.ndarray,
    lamda: float
) -> np.ndarray:
    '''
    Estimate the maximum likelihood values for the linear model

    Inputs :
    * Fi: [NxM] is the array of basis function vectors
    * t: [Nx1] is the target value for each input in Fi
    * lamda: The regularization constant

    Output: [Mx1], the maximum likelihood estimate of w for the linear model
    '''
    # y(x) = w^T * fi(x) = a^T * fi * fi(x)
    # a = w^T * fi = (K + lamda * I_N)^-1 * t
    # a.T*fi is the linear coefficients of the linear model
    # Gram matrix
    K = fi@fi.T
    
    a = np.linalg.inv(K + lamda*np.eye(K.shape[0]))@targets
    return a.T@fi

def linear_model(
    features: np.ndarray,
    mu: np.ndarray,
    sigma: float,
    w: np.ndarray
) -> np.ndarray:
    '''
    Inputs:
    * features: [NxD] is a data matrix with N D-dimensional data vectors.
    * mu: [MxD] matrix of M D dimensional mean vectors defining the
    multivariate normal distributions.
    * sigma: All normal distributions are isotropic with s*I covariance
    matrices (where I is the MxM identity matrix).
    * w: [Mx1] the weights, e.g. the output from the max_likelihood_linreg
    function.

    Output: [Nx1] The prediction for each data vector in features
    '''
    # y(x) = w * fi(x)
    # ath w is (10,) and fi(x) (Mvn_basis) is (N,M) (here N=150, M=10)
    return w@mvn_basis(features, mu, sigma).T


""" if __name__ == '__main__':
    X, t = load_regression_iris()
    N, D = X.shape
    print(X[:5, :])
    M, sigma = 10, 10
    mu = np.zeros((M, D))
    for i in range(D):
        mmin = np.min(X[i, :])
        mmax = np.max(X[i, :])
        mu[:, i] = np.linspace(mmin, mmax, M)
    print(mu)
    fi = mvn_basis(X, mu, sigma)
    #print(fi)
    #print(np.shape(fi))
    #_plot_mvn()
    lamda = 0.001
    wml = max_likelihood_linreg(fi, t, lamda)
    print(wml)
    prediction = linear_model(X, mu, sigma, wml)
    print(prediction)

    # plot actual values and predictions
    plt.plot(t, label='actual')
    plt.plot(prediction, label='prediction')
    plt.xlabel('data point')
    plt.ylabel('pedal length [cm]')
    plt.legend()
    plt.grid()
    plt.show()
    
    # plot the mean squared error
    # as a function of the feuature values
    plt.scatter((t - prediction)**2,t, color = 'red', s = 5)
    plt.xlabel('pedal length squared diffrence [cm^2]')
    plt.ylabel('pedal length [cm]')
    plt.grid()
    plt.show() """