# Author: Kristjan Orri Dadason
# Date: 02/09/2022
# Project: 04_classification
# Acknowledgements: tools.py and help.py from teacher
#


from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    return np.mean(features[targets==selected_class], axis=0)


def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    return np.cov(features[targets==selected_class],rowvar=False)


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    # maximum likelihood estimate
    return multivariate_normal(mean=class_mean, cov=class_covar).pdf(feature)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    for class_label in classes:
        means.append(mean_of_class(train_features, train_targets, class_label))
        covs.append(covar_of_class(train_features, train_targets, class_label))
    likelihoods = []
    for i in range(test_features.shape[0]):
        likelihoods.append([likelihood_of_class(test_features[i], means[j], covs[j]) for j in range(len(classes))])
    return np.array(likelihoods)


def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    return np.argmax(likelihoods, axis=1)


def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    #class conditional likelihoods
    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    #class priors
    prior_probabilies = []
    for class_label in classes:
        prior_probabilies.append(np.sum(train_targets==class_label)/len(train_targets))
    print(prior_probabilies)
    return likelihoods*np.array(prior_probabilies)
    
def confusion_matrix(predictions: np.ndarray, targets: np.ndarray, classes: np.ndarray):
    # sett upp eins og glaerum
    # https://en.wikipedia.org/wiki/Confusion_matrix
    # predicted values eru linur fylkis
    # actual values eru dalkar fylkis
    # svo hvert stak i fylki er [predicted, actual]
    conf_matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(predictions)):
        conf_matrix[test_targets[i], predictions[i]] += 1
    return conf_matrix

# independent section functions
# function that adds features to a class
# the new features are distributed normally around the mean of the class
def add_features(features, targets, class_to_add, std, num_to_add):
    # get mean of class
    mean = mean_of_class(features, targets, class_to_add)
    # add features to class
    features = np.concatenate((features, np.random.normal(mean, std, (num_to_add, features.shape[1]))), axis=0)
    return features

def add_targets(targets, class_to_add, num_to_add):
    targets = np.concatenate((targets, np.full(num_to_add, class_to_add)), axis=0)
    return targets

def plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list
):
    colors = ['red', 'purple', 'blue']
    # plot points each class in different color
    for class_label in classes:
        plt.scatter(points[point_targets==class_label,0], points[point_targets==class_label,2], color=colors[class_label], label=f'class {class_label}')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets)\
    = split_train_test(features, targets, train_ratio=0.6)
    # section 1.1
    #print(mean_of_class(train_features, train_targets, 0))
    # section 1.2
    #print(covar_of_class(train_features, train_targets, 0))
    # section 1.3
    #class_mean = mean_of_class(train_features, train_targets, 0)
    #class_cov = covar_of_class(train_features, train_targets, 0)
    #print(likelihood_of_class(test_features[0,:], class_mean, class_cov))
    # section 1.4
    likelihoods_max_like = maximum_likelihood(train_features, train_targets, test_features, classes)
    
    #print(likelihoods)
    # section 1.5
    #print(predict(likelihoods_max_like))

    """ # section 2.1
    likelihoods_max_aposteriori = maximum_aposteriori(train_features, train_targets, test_features, classes)
    #print(predict(likelihoods_max_aposteriori))

    # section 2.2
    # answer in pdf!!
    # 1.
    # compute accuracies
    max_lik=np.sum(predict(likelihoods_max_like)==test_targets)/len(test_targets)
    max_apost=np.sum(predict(likelihoods_max_aposteriori)==test_targets)/len(test_targets)
    print(f'maximum likelihood accuracy: {max_lik}')
    print(f'maximum a posteriori accuracy: {max_apost}')
    # accuracies are the same because the prior probabilities are the same for all classes
    # 2.
    # compute confusion matrix
    print('The confusion matrix for maximum likelihood is:')
    print(confusion_matrix(predict(likelihoods_max_like), test_targets, classes))
    print('The confusion matrix for maximum a posteriori is:')
    print(confusion_matrix(predict(likelihoods_max_aposteriori), test_targets, classes))
    # 3.
    # there is not a big difference between the confusion matrices
    # because the prior probabilities are the same for all classes """

    # independent section
    features, targets, classes = load_iris()
    features = add_features(features, targets, 0, 0.5, 2500)
    targets = add_targets(targets, 0, 2500)
    features = add_features(features, targets, 1, 0.8, 1000)
    targets = add_targets(targets, 1, 1000)
    (train_features, train_targets), (test_features, test_targets)\
    = split_train_test(features, targets, train_ratio=0.6)
    likelihoods_max_like = maximum_likelihood(train_features, train_targets, test_features, classes)
    likelihoods_max_aposteriori = maximum_aposteriori(train_features, train_targets, test_features, classes)
    # compute accuracies
    print(likelihoods_max_like[-1,:])
    print(likelihoods_max_aposteriori[-1,:])

    max_lik=np.sum(predict(likelihoods_max_like)==test_targets)/len(test_targets)
    max_apost=np.sum(predict(likelihoods_max_aposteriori)==test_targets)/len(test_targets)
    print(f'maximum likelihood accuracy: {max_lik}')
    print(f'maximum a posteriori accuracy: {max_apost}')

    # compute confusion matrix
    print('The confusion matrix for maximum likelihood is:')
    print(confusion_matrix(predict(likelihoods_max_like), test_targets, classes))
    print('The confusion matrix for maximum a posteriori is:')
    print(confusion_matrix(predict(likelihoods_max_aposteriori), test_targets, classes))

    print(features.shape)
    print(np.sum(targets==0)/len(targets))
    print(np.sum(targets==1)/len(targets))
    print(np.sum(targets==2)/len(targets))
    print(np.shape(likelihoods_max_aposteriori))
    
    plot_points(features, targets, classes)
    

    