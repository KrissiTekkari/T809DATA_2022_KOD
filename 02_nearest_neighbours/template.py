# Author: Kristjan Orri Dadason
# Date:
# Project: 
# Acknowledgements: plot points fra kennarar
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

if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    
    ## part 1
    
    d, t, classes = load_iris()
    x, points = d[0,:], d[1:, :]
    x_target, point_targets = t[0], t[1:]
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
    #print(knn_confusion_matrix(d_test, t_test, classes, 10))
    #print(knn_confusion_matrix(d_test, t_test, classes, 20))
    #print(best_k(d_train, t_train, classes))
    #knn_plot_points(d, t, classes, 3)

    # independent part
    #foo = k_nearest(x, points, 3)
    #print(point_targets[foo])
    #print(euclidian_distances(x, points)[foo])
    #print(weighted_vote(point_targets[foo], euclidian_distances(x, points)[foo], classes))
    #print(knn(x, points, point_targets, classes, 15))
    #print(wknn(x, points, point_targets, classes, 15))
    #predictions1 = knn_predict(d_test, t_test, classes, 10)
    #print(predictions1)
    #predictions1w = wknn_predict(d_test, t_test, classes, 10)
    #print(predictions1w)
    compare_knns(d_test, t_test, classes)

    # B. Theoretical

    # knn telur alla punkta sem jafn mikilvaega
    # svo thegar punktarnir sem eru langt i burtu eru taldir
    # med i k_nearest, tha ruglar thad reikniritid

    # hinsvegar thegar k haekkar tha naer weighted knn ad eyda 
    # "sudinu" fra punktunum sem eru fjaer. Thad gerist einmitt
    # af thvi ad 1/distance verdur svo litil tala og targets
    # langt i burtu detta i rauninni út.
    
    
    