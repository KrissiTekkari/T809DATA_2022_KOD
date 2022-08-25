# Author: Kristjan Orri Dadason
# Date: 19/08/2022
# Project: 02_nearest_neighbours
# Acknowledgements: plot points fra kennarar


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
    predic_acc = 0
    best_k = 0
    for i in range(len(points)-1):
        temp_predic_acc = knn_accuracy(points, point_targets, classes, i+1)
        if temp_predic_acc > predic_acc:
            predic_acc = temp_predic_acc
            best_k = i+1
    return best_k



def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    colors = ['yellow', 'purple', 'blue']
    target_prediction = knn_predict(points, point_targets, classes, k)
    for i in range(points.shape[0]):
        [x, y] = points[i,:2]
        if target_prediction[i] == point_targets[i]:
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='green',
                linewidths=2)
        else:
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='red',
                linewidths=2)
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.show()


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
    nearby_vigrar = np.zeros((len(classes),len(targets)))
    c_j = np.zeros(len(distances))
    for i in range(len(targets)):
        nearby_vigrar[targets[i],i] = 1
        # ef tvo features eru eins that kemur divide by zero error
        c_j[i] = 1/distances[i]
        
    
    c_j = (c_j/np.sum(c_j)).reshape(-1,1)
    check = nearby_vigrar@c_j
    check = np.sum(check, axis=1)

    return np.argmax(check)



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
    k_nearest_points = k_nearest(x, points, k)
    nearest_targets_to_point = point_targets[k_nearest_points]
    nearest_distances_to_point = euclidian_distances(x, points[k_nearest_points])
    return weighted_vote(nearest_targets_to_point, nearest_distances_to_point, classes)
    ...


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    prediction_listi = [0]*points.shape[0]
    
    for i in range(points.shape[0]):
        temp_points, temp_point_targets = remove_one(points, i), remove_one(point_targets, i)
        temp_prediction = wknn(points[i], temp_points, temp_point_targets, classes, k)
        prediction_listi[i] = temp_prediction
        
    return prediction_listi


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    knn_acc_array = np.zeros(len(points))
    wknn_acc_array = np.zeros(len(points))
    for i in range(len(points)):
        knn_target_prediction = knn_predict(points,targets,classes,i+1)
        wknn_target_prediction = wknn_predict(points,targets,classes,i+1)
        knn_array_true_if_correct_prediction = knn_target_prediction == targets
        wknn_array_true_if_correct_prediction = wknn_target_prediction == targets
        # True er 1, False er 0, telja nonzeros i predictions, deila med heildarfjolda predictions
        knn_acc_array[i] = np.count_nonzero(knn_array_true_if_correct_prediction)/len(knn_target_prediction)
        wknn_acc_array[i] = np.count_nonzero(wknn_array_true_if_correct_prediction)/len(wknn_target_prediction)
    
    plt.plot(np.arange(len(points)),knn_acc_array, label='knn')
    plt.plot(np.arange(len(points)),wknn_acc_array, label='wknn')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()