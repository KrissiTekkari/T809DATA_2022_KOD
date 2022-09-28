# Author: Kristjan Orri Dadason
# Date: 28/09/2022
# Project: 08_SVM
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


from unicodedata import name
from tools import plot_svm_margin, load_cancer
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np
import matplotlib.pyplot as plt


def _plot_linear_kernel():
    X, t = make_blobs(n_samples=40, centers=2, random_state=2)
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, t)
    plot_svm_margin(clf, X, t)


def _subplot_svm_margin(
    svc,
    X: np.ndarray,
    t: np.ndarray,
    num_plots: int,
    index: int
):
    '''
    Plots the decision boundary and decision margins
    for a dataset of features X and labels t and a support
    vector machine svc.

    Input arguments:
    * svc: An instance of sklearn.svm.SVC: a C-support Vector
    classification model
    * X: [N x f] array of features
    * t: [N] array of target labels
    '''
    # similar to tools.plot_svm_margin but added num_plots and
    # index where num_plots should be the total number of plots
    # and index is the index of the current plot being generated
    plt.subplot(1, num_plots, index)
    plot_svm_margin(svc, X, t)


def _compare_gamma():
    X, t = make_blobs(n_samples=40, centers=2, random_state=2)

    clf = svm.SVC(kernel='rbf', C=1000)
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 3, 1)
    plt.title('$\gamma$=default')
    
    clf = svm.SVC(kernel='rbf', C=1000, gamma = 0.2)
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 3, 2)
    plt.title('$\gamma=0.2$')

    clf = svm.SVC(kernel='rbf', C=1000, gamma = 2)
    clf.fit(X, t)
    _subplot_svm_margin(clf, X, t, 3, 3)
    plt.title('$\gamma=2$')


    plt.show()


def _compare_C():
    X, t = make_blobs(n_samples=40, centers=2, random_state=2)
    
    C_list = [1000, 0.5, 0.3, 0.05, 0.0001]
    for i in range(len(C_list)):
        clf = svm.SVC(kernel='linear', C=C_list[i])
        clf.fit(X, t)
        _subplot_svm_margin(clf, X, t, len(C_list), i+1)
        plt.title('$C$={}'.format(C_list[i]))
    plt.show()


def train_test_SVM(
    svc,
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray,
):
    '''
    Train a configured SVM on <X_train> and <t_train>
    and then measure accuracy, precision and recall on
    the test set

    This function should return (accuracy, precision, recall)
    '''
    svc.fit(X_train, t_train)
    accuracy = accuracy_score(t_test, svc.predict(X_test))
    precision = precision_score(t_test, svc.predict(X_test))
    recall = recall_score(t_test, svc.predict(X_test))
    return (accuracy, precision, recall)

if __name__ == '__main__':
    #_plot_linear_kernel()
    #_compare_gamma()
    #_compare_C()
    np.random.seed(1234)
    (X_train, t_train), (X_test, t_test) = load_cancer()
    
    svc_linear = svm.SVC(kernel='linear',C=1000)
    linear_kernel = train_test_SVM(svc_linear, X_train, t_train, X_test, t_test)
    
    svc_sigmoid = svm.SVC(kernel='sigmoid',C=1000)
    sigmoid_kernel = train_test_SVM(svc_sigmoid, X_train, t_train, X_test, t_test)
    
    svc_rbf = svm.SVC(kernel='rbf',C=1000)
    rbf_kernel = train_test_SVM(svc_rbf, X_train, t_train, X_test, t_test)
    
    svc_poly = svm.SVC(kernel='poly',C=1000)
    poly_kernel = train_test_SVM(svc_poly, X_train, t_train, X_test, t_test)
    
    print("                          accuracy           precision             recall")
    print('Linear kernel:     ', linear_kernel)
    print('Sigmoid kernel:    ', sigmoid_kernel)
    print('RBF kernel:        ', rbf_kernel)
    print('Polynomial kernel: ', poly_kernel)
    