# Author: Kristjan Orri Dadason
# Date: 12/10/2022
# Project: random_forests
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


from cgi import print_arguments
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score)

from collections import OrderedDict


class CancerClassifier:
    '''
    A general class to try out different sklearn classifiers
    on the cancer dataset
    '''
    def __init__(self, classifier, train_ratio: float = 0.7):
        self.classifier = classifier
        cancer = load_breast_cancer()
        self.X = cancer.data  # all feature vectors
        self.t = cancer.target  # all corresponding labels
        self.X_train, self.X_test, self.t_train, self.t_test =\
            train_test_split(
                cancer.data, cancer.target,
                test_size=1-train_ratio, random_state=109)

        # Fit the classifier to the training data here
        self.classifier.fit(self.X_train, self.t_train)
        self.predictions = self.classifier.predict(self.X_test)

    def confusion_matrix(self) -> np.ndarray:
        '''Returns the confusion matrix on the test data
        '''
        # 0: malignant ("positive"), 1: benign ("negative")
        # confusion matrix set up like this:
        # https://en.wikipedia.org/wiki/Confusion_matrix
        #                               prediction
        #                      malignant          benign
        #         malignant | True Positive  | False Negative | recall = TP/(TP+FN)
        # actual  benign    | False Positive | True Negative  |
        #         precision = TP/(TP+FP)                       accuracy = (TP+TN)/(TP+TN+FP+FN)
    
        num_classes = len(np.unique(self.t))
        conf_matrix = np.zeros((num_classes, num_classes))
        for i in range(len(self.predictions)):
            conf_matrix[self.t_test[i], self.predictions[i]] += 1
        return conf_matrix

    def accuracy(self) -> float:
        '''Returns the accuracy on the test data
        '''
        return accuracy_score(self.t_test, self.predictions)

    def precision(self) -> float:
        '''Returns the precision on the test data
        '''
        # I put pos_label = 0, because malignant is the "positive" class
        return precision_score(self.t_test, self.predictions, pos_label = 0)

    def recall(self) -> float:
        '''Returns the recall on the test data
        '''
        # I put pos_label = 0, because malignant is the "positive" class
        # Higher recall means less false negatives, which is good
        return recall_score(self.t_test, self.predictions, pos_label = 0)

    def cross_validation_accuracy(self) -> float:
        '''Returns the average 10-fold cross validation
        accuracy on the entire dataset.
        '''
        return np.mean(cross_val_score(self.classifier, self.X, self.t, cv=10))

    def feature_importance(self) -> list:
        '''
        Draw and show a barplot of feature importances
        for the current classifier and return a list of
        indices, sorted by feature importance (high to low).
        '''
        ...


def _plot_oob_error():
    RANDOM_STATE = 1337
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features='log2',
                oob_score=True,
                random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(
                n_estimators=100,
                warm_start=True,
                max_features=None,
                oob_score=True,
                random_state=RANDOM_STATE))]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 30
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(...,  ...)  # Use cancer data here
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


def _plot_extreme_oob_error():
    ...


# main function
if __name__ == '__main__':
    print("PROGRAM START\n")
    
    """ cancer = load_breast_cancer()
    X = cancer.data  # all feature vectors
    t = cancer.target  # all corresponding labels
    
    print("Number of 1's in t: ", np.sum(t)) """
    
    classifier_type = sklearn.tree.DecisionTreeClassifier()
    cc = CancerClassifier(classifier = classifier_type)
    conf_mat = cc.confusion_matrix()
    print(f"Confusion matrix:      [{conf_mat[0,:]}")
    print(f"                       {conf_mat[1,:]}]")
    print("                  Accuracy:", np.round(cc.accuracy(),4))
    print("                 Precision:", np.round(cc.precision(),4))
    print("                    Recall:", np.round(cc.recall(),4))
    print("Cross validation accuracy: ", np.round(cc.cross_validation_accuracy(),4))