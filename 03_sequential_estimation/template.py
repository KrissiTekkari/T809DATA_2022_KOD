# Author: Kristjan Orri Dadason
# Date: 31/08/2022
# Project: Sequential Estimation
# Acknowledgements: 
#


from tools import scatter_3d_data, bar_per_axis

import matplotlib.pyplot as plt
import numpy as np


def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    I_k = np.eye(k)
    return np.random.multivariate_normal(mean, (var**2)*I_k, n)


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    return mu + (x - mu)/n

def _plot_sequence_estimate():
    data = gen_data(100, 3, np.array([0, 0, 0]), np.sqrt(3))
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        estimates.append(update_sequence_mean(estimates[-1], data[i], i+1))
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    return np.mean(((y - y_hat)**2))


def _plot_mean_square_error():
    # plot mean square error between estimate and actual mean
    actual_mean = np.array([0, 0, 0])
    data = gen_data(100, 3, actual_mean, np.sqrt(3))
    estimates = [np.array([0, 0, 0])]
    SE = []
    for i in range(data.shape[0]):
        new_estimate = update_sequence_mean(estimates[-1], data[i], i+1)
        estimates.append(new_estimate)
        SE.append(_square_error(actual_mean, new_estimate))
    plt.plot(SE)
    plt.show()


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # remove this if you don't go for the independent section
    ...


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    ...

# main function
if __name__ == '__main__':
    # 1.1
    #np.random.seed(1234)
    #print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
    #np.random.seed(1234)
    #print(gen_data(5, 1, np.array([0.5]), 0.5))
    # 1.2
    #np.random.seed(1234)
    #X = gen_data(300, 3, np.array([0, 1, -1]), np.sqrt(3))
    #scatter_3d_data(X)
    #bar_per_axis(X)
    # 1.4
    #mean = np.mean(X, 0)
    #new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
    #print(update_sequence_mean(mean, new_x, X.shape[0]))
    # 1.5
    #_plot_sequence_estimate()
    #_plot_mean_square_error()
    # independent section
    print('lol')