# Author: Kristjan Orri Dadason
# Date:
# Project: 
# Acknowledgements: 
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree

from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
    '''
    Calculate the prior probability of each class type
    given a list of all targets and all class types
    '''
    N = len(targets)
    temp_list = [0] * len(classes)
    for i in classes:
        k = 0
        for j in targets:
            if j == i:
                k += 1
        temp_list[i] = k / N
    return np.array(temp_list)


def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    features = features[:, split_feature_index]
    
    features_1 = features[features < theta]
    targets_1 = targets[features < theta]

    features_2 = features[features > theta]
    targets_2 = targets[features > theta]

    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    return 1/2 * (1 - sum(np.power(prior(targets, classes),2)))
    ...


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    n = t1.shape[0] + t2.shape[0]
    return t1.shape[0]*g1/n + t2.shape[0]*g2/n


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    Calculate the gini impurity for a split on split_feature_index
    for a given dataset of features and targets.
    '''
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    return weighted_impurity(t_1, t_2, classes)
    ...


def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        # fara i feature dalk, fara fra laegsta i haesta gildi med num_tries bilum (excluding max og min)
        thetas = np.linspace(features[:, i].min(), features[:, i].max(), num_tries+2)[1:-1]
        # iterate thresholds
        for theta in thetas:
            gini = total_gini_impurity(features, targets, classes, i, theta)
            if gini < best_gini:
                best_gini = gini
                best_dim = i
                best_theta = theta
            
    return best_gini, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        self.tree = self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        # check model on test data
        #target_prediction = self.tree.predict(self.test_features)
        target_prediction = self.guess()
        array_true_if_correct_prediction = target_prediction == self.test_targets
        # True er 1, False er 0, telja nonzeros i predictions, deila med heildarfjolda predictions
        return np.count_nonzero(array_true_if_correct_prediction)/len(target_prediction)

    def plot(self):
        plot_tree(self.tree, filled=True)
        plt.show()

    def plot_progress(self):
        # Independent section
        # Remove this method if you don't go for independent section.

        num_train_samples = len(self.train_features)
        # accuracy sem fer a y-as
        accuracies = [0]*num_train_samples
        # iterate through training features and targets
        # tek fyrst bara 1 (efsta), svo 2 (efstu), svo 3......
        # thangad til tek allt i einu
        # fit tree i hvert skipti
        # finna accuracy i hvert skipti
        
        # EDA, taka random indices i hvert skipti (til ad nota ef gogn eru ekki randomly rodud upp)
        # s.s. velja fyrst 1 random feature, fitta, svo 2 random feature , fitta o.s.frv.
        # replace = False svo sama feature er ekki tekid oftar en einu sinni (i np.random.choice)
        # ahugaverdur munur a ad nota random indices vs ekki random (sja plots)
        # hradar ad laera af faum samples med random!
        
        for i in range(num_train_samples):
            ######## nota ef gogn eru ekki randomly rodud upp ##############
            #random_indices = np.random.choice(num_train_samples, i+1, replace=False)
            #self.tree = self.tree.fit(self.train_features[random_indices], self.train_targets[random_indices])
            # ------------------------------------------------------------------------------------------------ #

            self.tree.fit(self.train_features[:(i+1)], self.train_targets[:(i+1)])
            
            accuracies[i]=self.accuracy()
        
        plt.plot(np.arange(num_train_samples),accuracies)
        plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy')
        plt.show()
            

    def guess(self):
        return self.tree.predict(self.test_features)

    def confusion_matrix(self):
        # sett upp eins og i wikipedia greininni
        # https://en.wikipedia.org/wiki/Confusion_matrix
        # predicted values eru linur fylkis
        # actual values eru dalkar fylkis
        # svo hvert stak i fylki er [actual, predicted]
        target_prediction = self.tree.predict(self.test_features)
        conf_matrix = np.zeros((3, 3))
        for i in range(len(target_prediction)):
            conf_matrix[self.test_targets[i], target_prediction[i]] += 1
        return conf_matrix

if __name__ == '__main__':
    # select your function to test here and do `python3 template.py`
    #print(prior([0, 0, 1], [0, 1]))
    #print(prior([0, 2, 3, 3], [0, 1, 2, 3]))
    #split_feature_index = 2
    #theta = 4.65
    features, targets, classes = load_iris()
    #(f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    #print(f_1.shape, f_2.shape)
    #print(gini_impurity(t_1, classes))
    #print(gini_impurity(t_2, classes))
    #print(weighted_impurity(t_1, t_2, classes))
    #print(total_gini_impurity(features, targets, classes, 2, 4.65))
    #print(brute_best_split(features, targets, classes, 30))
    dt = IrisTreeTrainer(features, targets, classes=classes, train_ratio=0.6)
    dt.train()
    #print(dt.accuracy())
    #print(dt.guess())
    #print(dt.confusion_matrix())
    dt.plot_progress()