# Author: Kristjan Orri Dadason
# Date: 28/10/2022
# Project: k_means
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.



import numpy as np
import sklearn as sk
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from typing import Union

from tools import load_iris, image_to_numpy, plot_gmm_results


def distance_matrix(
    X: np.ndarray,
    Mu: np.ndarray
) -> np.ndarray:
    '''
    Returns a matrix of euclidian distances between points in
    X and Mu.

    Input arguments:
    * X (np.ndarray): A [n x f] array of samples
    * Mu (np.ndarray): A [k x f] array of prototypes

    Returns:
    out (np.ndarray): A [n x k] array of euclidian distances
    where out[i, j] is the euclidian distance between X[i, :]
    and Mu[j, :]
    '''
    (n,f) = np.shape(X)
    (k,f) = np.shape(Mu)
    dist_matrix = np.zeros((n,k))
    for i in range(n):
        D = X[i,:] - Mu
        dist_matrix[i,:] = np.linalg.norm(D, axis=1)
    return dist_matrix
        


def determine_r(dist: np.ndarray) -> np.ndarray:
    '''
    Returns a matrix of binary indicators, determining
    assignment of samples to prototypes.

    Input arguments:
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    out (np.ndarray): A [n x k] array where out[i, j] is
    1 if sample i is closest to prototype j and 0 otherwise.
    '''
    (n,k) = np.shape(dist)
    determ = np.zeros((n,k))
    where_min = np.argmin(dist, axis=1)
    for i in range(n):
        determ[i,where_min[i]] = 1
    return determ


def determine_j(R: np.ndarray, dist: np.ndarray) -> float:
    '''
    Calculates the value of the objective function given
    arrays of indicators and distances.

    Input arguments:
    * R (np.ndarray): A [n x k] array where out[i, j] is
        1 if sample i is closest to prototype j and 0
        otherwise.
    * dist (np.ndarray): A [n x k] array of distances

    Returns:
    * out (float): The value of the objective function
    '''
    (n,k) = np.shape(R)
    s = R*dist
    return np.sum(s)/n


def update_Mu(
    Mu: np.ndarray,
    X: np.ndarray,
    R: np.ndarray
) -> np.ndarray:
    '''
    Updates the prototypes, given arrays of current
    prototypes, samples and indicators.

    Input arguments:
    Mu (np.ndarray): A [k x f] array of current prototypes.
    X (np.ndarray): A [n x f] array of samples.
    R (np.ndarray): A [n x k] array of indicators.

    Returns:
    out (np.ndarray): A [k x f] array of updated prototypes.
    '''
    (n,f) = np.shape(X)
    (k,f) = np.shape(Mu)
    for i in range(k):
        Mu[i,:] = np.sum(R[:,i].reshape(n,1)*X, axis=0)/np.sum(R[:,i])
    return Mu


def k_means(
    X: np.ndarray,
    k: int,
    num_its: int
) -> Union[list, np.ndarray, np.ndarray]:
    # We first have to standardize the samples
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standard = (X-X_mean)/X_std
    # run the k_means algorithm on X_st, not X.

    # we pick K random samples from X as prototypes
    nn = sk.utils.shuffle(range(X_standard.shape[0]))
    Mu = X_standard[nn[0: k], :]
    
    J_list = []
    # run for num_its iterations
    for i in range(num_its):
        dist = distance_matrix(X_standard, Mu)
        R = determine_r(dist)
        J_list.append(determine_j(R, dist))
        Mu = update_Mu(Mu, X_standard, R)

    # Then we have to "de-standardize" the prototypes
    for i in range(k):
        Mu[i, :] = Mu[i, :] * X_std + X_mean
    return Mu, R, J_list



def _plot_j():
    X, y, c = load_iris()
    _, _, J = k_means(X, 4, 10)
    plt.plot(J)
    plt.xlabel('Iterations')
    plt.ylabel('Objective function value')
    plt.show()


def _plot_multi_j():
    X, y, c = load_iris()
    J_list = []
    runs = 4
    k = (2,3,5,10)
    for i in range(runs):
        _, _, J = k_means(X, k[i], 10)
        J_list.append(J)
    # plot the J_list on the same plot
    for i in range(runs):
        plt.plot(J_list[i], label='k = {}'.format(k[i]))
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Objective function value')
    plt.show()


def k_means_predict(
    X: np.ndarray,
    t: np.ndarray,
    classes: list,
    num_its: int
) -> np.ndarray:
    '''
    Determine the accuracy and confusion matrix
    of predictions made by k_means on a dataset
    [X, t] where we assume the most common cluster
    for a class label corresponds to that label.

    Input arguments:
    * X (np.ndarray): A [n x f] array of input features
    * t (np.ndarray): A [n] array of target labels
    * classes (list): A list of possible target labels
    * num_its (int): Number of k_means iterations

    Returns:
    * the predictions (list)
    '''
    _, R, _= k_means(X, len(classes), num_its)
    clusters = []
    for i in range(len(classes)):
        check = np.argmax(R[t == classes[i], :], axis=1)
        count = np.bincount(check)
        cluster = np.argmax(count)
        clusters.append(cluster)

    predictions = []
    for i in range(np.shape(R)[0]):
        predictions.append(classes[clusters[np.argmax(R[i, :])]])
    return predictions


def _iris_kmeans_accuracy():
    X, y, c = load_iris()
    prediction = k_means_predict(X, y, c, 5)
    acc = accuracy_score(y, prediction)
    conf_matrix = sk.metrics.confusion_matrix(y, prediction)
    print('Accuracy: {}'.format(acc))
    print('Confusion matrix:')
    print(conf_matrix)

def _my_kmeans_on_image():
    image, (w, h) = image_to_numpy()
    k_means(image, 7, 5)


def plot_image_clusters(n_clusters: int):
    '''
    Plot the clusters found using sklearn k-means.
    '''
    image, (w, h) = image_to_numpy()
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(image)
    plt.subplot(121)
    plt.imshow(image.reshape(w, h, 3))
    plt.subplot(122)
    # uncomment the following line to run
    plt.imshow(kmeans.labels_.reshape(w, h), cmap="plasma")
    plt.show()


def _gmm_info():
    X, y, c = load_iris()
    gauss = GaussianMixture(n_components=3)
    # fit the model
    gauss.fit(X)
    print('Mixing coefficients:')
    print(gauss.weights_)
    print('Mean vectors:')
    print(gauss.means_)
    print('Covariance matrices:')
    print(gauss.covariances_)
    


def _plot_gmm():
    X, y, c = load_iris()
    gauss = GaussianMixture(n_components=3)
    # fit the model
    gauss.fit(X)
    # make predictions
    predictions = gauss.predict(X)
    plot_gmm_results(X, predictions, gauss.means_, gauss.covariances_)
    

if __name__ == '__main__':
    """ a = np.array([
        [1, 0, 0],
        [4, 4, 4],
        [2, 2, 2]])
    b = np.array([
        [0, 0, 0],
        [4, 4, 4]])
    print(distance_matrix(a, b))
    
    dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])
    
    print(determine_r(dist))
    
    dist = np.array([
        [  1,   2,   3],
        [0.3, 0.1, 0.2],
        [  7,  18,   2],
        [  2, 0.5,   7]])
    R = determine_r(dist)
    print(determine_j(R, dist))
    
    
    X = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0]])
    Mu = np.array([
    [0.0, 0.5, 0.1],
    [0.8, 0.2, 0.3]])
    R = np.array([
    [1, 0],
    [0, 1],
    [1, 0]])
    print(update_Mu(Mu, X, R)) """
    #_plot_j()
    #_plot_multi_j()
    #X, y, c = load_iris()
    #predict = k_means_predict(X, y, c, 5)
    #print(predict)
    #_iris_kmeans_accuracy()
    """ num_clusters = [2,5,10,20]
    for i in range(len(num_clusters)):
        plot_image_clusters(num_clusters[i]) """
    #_gmm_info()
    _plot_gmm()

        