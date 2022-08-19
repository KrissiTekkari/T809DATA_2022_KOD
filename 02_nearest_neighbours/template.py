# Author: Kristjan Orri Dadason
# Date:
# Project: 
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points
from help import remove_one


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    # norm sama og lengd vigurs 
    return np.linalg.norm(x - y)


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])
    return distances


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    distances = euclidian_distances(x, points)
    # notum np.argsort fra hint
    # thad gefur indexes sem myndi sorta fra laegsta til haesta gildi
    # slice-um svo k fyrstu ut og returnum thvi
    return np.argsort(distances)[:k]


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # telja classes med np.bincount
    # count = [fjoldi target 0, fjoldi target 1, fjoldi target 2]
    count = np.bincount(targets) 
    # skila svo ut max gilda indexinu 
    # sem er tha sa klasi sem er mest af i kringum punkt
    return np.argmax(count)


def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    nearest_classes_to_point = point_targets[k_nearest(x, points, k)]
    return vote(nearest_classes_to_point, classes)


def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    
    prediction_listi = [0]*points.shape[0]
    
    for i in range(points.shape[0]):
        temp_points, temp_point_targets = remove_one(points, i), remove_one(point_targets, i)
        temp_prediction = knn(points[i], temp_points, temp_point_targets, classes, k)
        prediction_listi[i] = temp_prediction
        
    return prediction_listi


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    # check model on test data
    target_prediction = knn_predict(points,point_targets,classes,k)
    array_true_if_correct_prediction = target_prediction == point_targets
    # True er 1, False er 0, telja nonzeros i predictions, deila med heildarfjolda predictions
    return np.count_nonzero(array_true_if_correct_prediction)/len(target_prediction)


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # sett upp eins og glaerum
    # https://en.wikipedia.org/wiki/Confusion_matrix
    # predicted values eru linur fylkis
    # actual values eru dalkar fylkis
    # svo hvert stak i fylki er [predicted, actual]
    target_prediction = knn_predict(points,point_targets,classes,k)
    conf_matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(target_prediction)):
        conf_matrix[target_prediction[i], point_targets[i]] += 1
    return conf_matrix


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    ...


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    ...


def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # Remove if you don't go for independent section
    ...


def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    # Remove if you don't go for independent section
    ...


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    ...


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section
    ...

if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    
    ## part 1
    
    #d, t, classes = load_iris()
    #x, points = d[0,:], d[1:, :]
    #x_target, point_targets = t[0], t[1:]
    #print(euclidian_distance(x, points[0]))
    #print(euclidian_distance(x, points[50]))
    #print(euclidian_distances(x, points))
    #print(k_nearest(x, points, 3))
    #print(vote(np.array([0,0,1,2]), np.array([0,1,2])))
    #print(vote(np.array([1,1,1,1]), np.array([0,1])))
    
    #print(x_target)
    #print(knn(x, points, point_targets, classes, 1))
    #print(knn(x, points, point_targets, classes, 15))
    #print(knn(x, points, point_targets, classes, 150))
    
    ## part 2
    
    d, t, classes = load_iris()
    (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)
    #predictions1 = knn_predict(d_test, t_test, classes, 10)
    #print(predictions1)
    #predictions2 = knn_predict(d_test, t_test, classes, 5)
    #print(predictions2)
    #print(knn_accuracy(d_test, t_test, classes, 10))
    #print(knn_accuracy(d_test, t_test, classes, 5))
    print(knn_confusion_matrix(d_test, t_test, classes, 10))
    print(knn_confusion_matrix(d_test, t_test, classes, 20))

    
    