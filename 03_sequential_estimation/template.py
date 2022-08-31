# Author: 
# Date:
# Project: 
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
    ...


def _plot_sequence_estimate():
    data = ...
    estimates = [np.array([0, 0, 0])]
    for i in range(data.shape[0]):
        ...
    plt.plot([e[0] for e in estimates], label='First dimension')
    ...
    plt.legend(loc='upper center')
    plt.show()


def _square_error(y, y_hat):
    ...


def _plot_mean_square_error():
    ...


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
    np.random.seed(1234)
    print(gen_data(2, 3, np.array([0, 1, -1]), 1.3))
    np.random.seed(1234)
    print(gen_data(5, 1, np.array([0.5]), 0.5))