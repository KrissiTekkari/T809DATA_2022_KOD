from typing import Union
import numpy as np

from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if x < -100:
        return 0
    return 1/(1 + np.exp(-x))


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    return sigmoid(x)*(1 - sigmoid(x))


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    weighted_sum = np.dot(x, w)
    activation = sigmoid(weighted_sum)
    return weighted_sum, activation


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    y = np.zeros(K)
    
    # add 1 to the beggining of "input vectors"
    z0 = np.insert(x, 0, 1)
    z1 = np.insert(np.zeros(M), 0, 1)
    
    a1 = np.zeros(M)
    a2 = np.zeros(K)
    
    for i in range(M):
        a1[i], z1[i+1] = perceptron(z0, W1[:, i])
    for i in range(K):
        a2[i], y[i]  = perceptron(z1, W2[:, i])

    return y, z0, z1, a1, a2
    


def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair  x, target_y
    '''
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    d_k = y - target_y
    d_j = np.zeros(M)
    for j in range(M):
        #for k in range(K):
        #    d_j[j] += d_k[k]*W2[j+1,k]
        # OR this, I like it more ;)
        d_j[j] = np.dot(d_k, W2[j+1, :])
        d_j[j] *= d_sigmoid(a1[j])
    
    dE1 = np.outer(z0, d_j)
    dE2 = np.outer(z1, d_k)
    
    return y, dE1, dE2
    
    


def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Train a network by:
    1. forward propagating an input feature through the network
    2. Calculate the error between the prediction the network
    made and the actual target
    3. Backpropagating the error through the network to adjust
    the weights.
    '''
    N = len(X_train)
    # one hot train targets
    t_train = np.eye(K)[t_train]
    
    E_total = np.zeros(iterations)
    misclassification_rate = np.zeros(iterations)

    for i in range(iterations):
        dE1_total = np.zeros(np.shape(W1))
        dE2_total = np.zeros(np.shape(W2))
        y_all = np.zeros((N, K))
        for j in range(N):
            y, dE1, dE2 = backprop(X_train[j], t_train[j], M, K, W1, W2)
            y_all[j] = y
            dE1_total += dE1
            dE2_total += dE2
            E_total[i] += -(t_train[j]@np.log(y) + (1-t_train[j])@np.log(1-y))
        W1 = W1 - eta*dE1_total/N
        W2 = W2 - eta*dE2_total/N
        misclassification_rate[i] = np.sum(np.argmax(y_all, axis=1) != np.argmax(t_train, axis=1))/N

    return W1, W2, E_total, misclassification_rate, np.argmax(y_all, axis=1)


def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    N = len(X)
    y_all = np.zeros((N, K))
    for j in range(N):
        y, z0, z1, a1, a2 = ffnn(X[j], M, K, W1, W2)
        y_all[j] = y
    return np.argmax(y_all, axis=1)

def confusion_matrix(predictions: np.ndarray, targets: np.ndarray, classes: np.ndarray):
    # sett upp eins og glaerum
    # https://en.wikipedia.org/wiki/Confusion_matrix
    # predicted values eru linur fylkis
    # actual values eru dalkar fylkis
    # svo hvert stak i fylki er [predicted, actual]
    conf_matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(predictions)):
        conf_matrix[targets[i], predictions[i]] += 1
    return conf_matrix


def E_total_plot(x,y):
    plt.plot(x,y, label = "Training error", color='red')
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.grid()
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.show()

def misclassification_rate_plot(x,y):
    plt.plot(x,y, label = "Misclassification rate", color='blue')
    plt.xlabel("Iterations")
    plt.ylabel("Misclassification rate")
    plt.grid()
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.show()
    
# independent section functions
def plot_this(x,y, title, xlabel, ylabel):
    plt.plot(x,y, label = "Misclassification rate", color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.gca().yaxis.set_minor_locator(AutoMinorLocator())
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlim(xmin=0)
    #plt.ylim(ymin=0)
    plt.show()
    

if __name__ == '__main__':
    ############################## Section 1 ################################
    # section 1.1
    #print(sigmoid(0.5))
    #print(d_sigmoid(0.2))
    # section 1.2
    #print(perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1])))
    #print(perceptron(np.array([0.2,0.4]),np.array([0.1,0.4])))
    # section 1.3
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = \
    split_train_test(features, targets)
    
    """ # initialize the random generator to get repeatable results
    np.random.seed(1234)
    # Take one point:
    x = train_features[0, :]
    K = 3 # number of classes
    M = 10
    D = 4
    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    print(f'y: {y}')
    print(f'z0: {z0}')
    print(f'z1: {z1}')
    print(f'a1: {a1}')
    print(f'a2: {a2}') """
    """ # initialize random generator to get predictable results
    np.random.seed(42)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    x = features[0, :]

    # create one-hot target for the feature
    target_y = np.zeros(K)
    target_y[targets[0]] = 1.0

    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1

    y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
    print(f'y: {y}')
    print(f'dE1: {dE1}')
    print(f'dE2: {dE2}') """
    
    ############################## Section 2 ################################
    # section 2.1
    # initialize the random seed to get predictable results
    """ np.random.seed(1234)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
        train_features[:20, :], train_targets[:20], M, K, W1, W2, 1000, 0.1)
    
    print(f'W1tr: {W1tr}')
    print(f'W2tr: {W2tr}')
    print(f'Etotal: {Etotal}')
    print(f'misclassification_rate: {misclassification_rate}')
    print(f'last_guesses: {last_guesses}') """
    
    # section 2.2
    #guesses = test_nn(test_features, M, K, W1tr, W2tr)
    
    
    # section 2.3
    """ np.random.seed(1234)

    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]

    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    iterations = 500
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
        train_features, train_targets, M, K, W1, W2, iterations, 0.1)
    
    guesses = test_nn(test_features, M, K, W1tr, W2tr)
    # 1.
    accuracy = np.sum(guesses == test_targets)/len(test_targets)
    print(f'Accuracy: {accuracy}')
    # 2.
    conf_matrix = confusion_matrix(guesses, test_targets, classes)
    print(conf_matrix)
    # 3.
    E_total_plot(range(iterations), Etotal)
    # 4.
    misclassification_rate_plot(range(iterations), misclassification_rate) """
    
    # Independent section

    # Initialize two random weight matrices
    #M = 6  # number of hidden units
    D = train_features.shape[1]
    iterations = 500
    K = 3  # number of classes
    """ iii = 10
    accuracy_array = np.zeros(iii)
    
    for i in range(1, iii+1):
        np.random.seed(1234)
        M = i
        W1 = 2 * np.random.rand(D + 1, M) - 1
        W2 = 2 * np.random.rand(M + 1, K) - 1
        W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
            train_features, train_targets, i, K, W1, W2, iterations, 0.1)
        guesses = test_nn(test_features, M, K, W1tr, W2tr)
        accuracy = np.sum(guesses == test_targets)/len(test_targets)
        accuracy_array[i-1] = accuracy
    
    # plot the accuracy as a function of the number of hidden units
    plot_this(range(1, iii+1), accuracy_array,'Accuracy as function of hidden units', 
              'Number of hidden units', 'Accuracy') """
    
    """ jjj = 10
    accuracy_eta_array = np.zeros(jjj)
    eta_array = np.zeros(jjj)
    for i in range(1, jjj+1):
        np.random.seed(1234)
        M = 6
        W1 = 2 * np.random.rand(D + 1, M) - 1
        W2 = 2 * np.random.rand(M + 1, K) - 1
        W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
            train_features, train_targets, M, K, W1, W2, iterations, 0.1*i)
        guesses = test_nn(test_features, M, K, W1tr, W2tr)
        accuracy = np.sum(guesses == test_targets)/len(test_targets)
        accuracy_eta_array[i-1] = accuracy
        eta_array[i-1] = 0.1*i
    plot_this(eta_array, accuracy_eta_array,'Accuracy as function of learning rate', 
              '$\eta$', 'Accuracy') """
    
    kkk = 10
    accuracy_eta_array = np.zeros(kkk)
    eta_array = np.zeros(kkk)
    for i in range(1, kkk+1):
        np.random.seed(1234)
        M = 6
        W1 = 2 * np.random.rand(D + 1, M) - 1
        W2 = 2 * np.random.rand(M + 1, K) - 1
        W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(
            train_features, train_targets, M, K, W1, W2, iterations, 0.1*i)
        guesses = test_nn(test_features, M, K, W1tr, W2tr)
        accuracy = np.sum(guesses == test_targets)/len(test_targets)
        accuracy_eta_array[i-1] = accuracy
        eta_array[i-1] = 0.1*i
    plot_this(eta_array, accuracy_eta_array,'Accuracy as function of learning rate', 
              '$\eta$', 'Accuracy')