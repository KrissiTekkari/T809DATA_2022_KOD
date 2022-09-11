# Author: Kristjan Orri Dadason
# Date: 31/08/2022
# Project: Sequential Estimation
# Acknowledgements: 
#


from cmath import inf
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
    I_k = np.eye(k)
    changing_data_array = np.zeros((n, k))
    for i in range(n):
        changing_data_array[i] = np.random.multivariate_normal(start_mean, (var**2)*I_k)
        start_mean = start_mean + (end_mean - start_mean)/n	
    return changing_data_array


def _plot_changing_sequence_estimate():
    def update_sequence_changing_mean(
        mu: np.ndarray,
        x: np.ndarray,
        alpha: int
    ) -> np.ndarray:
        '''Performs the mean sequence estimation update on a changing mean
        '''
        return mu + (x - mu)*alpha
    
    def _plot_changing_mean_square_error(data,estimates,start_mean,end_mean):
        # plot mean square error between estimate and actual mean
        actual_mean = start_mean
        SE = []
        for i in range(data.shape[0]):
            SE.append(_square_error(actual_mean, estimates[i]))
            actual_mean =  actual_mean + (end_mean -  actual_mean)/data.shape[0]
        plt.plot(SE)
        print(np.sum(SE))
        plt.show()
        
    def sum_mean_square_error(data,estimates,start_mean,end_mean):
        # plot mean square error between estimate and actual mean
        actual_mean = start_mean
        SE = []
        for i in range(data.shape[0]):
            SE.append(_square_error(actual_mean, estimates[i]))
            actual_mean =  actual_mean + (end_mean -  actual_mean)/data.shape[0]
        return np.sum(SE)
    # remove this if you don't go for the independent section
    N = 500
    start_mean = np.array([0, 1, -1])
    end_mean = np.array([1, -1, 0])
    data = gen_changing_data(N, 3, start_mean, end_mean, np.sqrt(3))
    """ estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        update = update_sequence_mean(estimates[-1], data[i], i+1)
        estimates.append(update) """
    # find alpha that minimizes the sum of mean square error
    best_alpha = 0
    best_MSE = inf
    best_estimates = []
    # try alphas from 0 to 1 with 0.005 step size
    for alpha in np.linspace(0,1,200):
        estimates = [np.array([0, 0, 0])]
        for i in range(data.shape[0]):
            update = update_sequence_changing_mean(estimates[-1], data[i], alpha)
            estimates.append(update)
        MSE = sum_mean_square_error(data,estimates,start_mean,end_mean)
        #print(alpha,MSE)
        if MSE < best_MSE:
            best_MSE = MSE
            best_alpha = alpha
            best_estimates = estimates
    print(best_alpha,best_MSE)
    plt.plot([e[0] for e in best_estimates], label='First dimension', color='blue')
    plt.plot([e[1] for e in best_estimates], label='Second dimension', color='red')
    plt.plot([e[2] for e in best_estimates], label='Third dimension', color='green')
    # plot lines that start from the start mean and end at the end mean
    plt.plot([start_mean[0] + (end_mean[0] - start_mean[0])/N*i for i in range(N)], color='blue', linestyle='dashed')
    plt.plot([start_mean[1] + (end_mean[1] - start_mean[1])/N*i for i in range(N)], color='red', linestyle='dashed')
    plt.plot([start_mean[2] + (end_mean[2] - start_mean[2])/N*i for i in range(N)], color='green', linestyle='dashed')
    plt.legend(loc='upper center')
    plt.grid()
    plt.show()
    _plot_changing_mean_square_error(data,best_estimates,start_mean,end_mean)

# main function
if __name__ == '__main__':
    # 1.1
    #np.random.seed(1234)
    print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
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
    #print('lol')
    #Y = gen_changing_data(500, 3, np.array([0, 1, -1]), np.array([1, -1, 0]), np.sqrt(3))
    #scatter_3d_data(Y)
    #bar_per_axis(Y)
    _plot_changing_sequence_estimate()

    