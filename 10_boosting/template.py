# Author: Kristjan Orri Dadason
# Date: 15/10/2022
# Project: Boosting
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import impute

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score)

from tools import get_titanic, build_kaggle_submission


def get_better_titanic():
    '''
    Loads the cleaned titanic dataset but change
    how we handle the age column.
    '''
    '''
    Loads the cleaned titanic dataset
    '''

    # Load in the raw data
    # check if data directory exists for Mimir submissions
    # DO NOT REMOVE
    if os.path.exists('./data/train.csv'):
        train = pd.read_csv('./data/train.csv')
        test = pd.read_csv('./data/test.csv')
    else:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')

    # Concatenate the train and test set into a single dataframe
    # we drop the `Survived` column from the train set
    X_full = pd.concat([train.drop('Survived', axis=1), test], axis=0)

    # The cabin category consist of a letter and a number.
    # We can divide the cabin category by extracting the first
    # letter and use that to create a new category. So before we
    # drop the `Cabin` column we extract these values
    X_full['Cabin_mapped'] = X_full['Cabin'].astype(str).str[0]
    # Then we transform the letters into numbers
    cabin_dict = {k: i for i, k in enumerate(X_full.Cabin_mapped.unique())}
    X_full.loc[:, 'Cabin_mapped'] =\
        X_full.loc[:, 'Cabin_mapped'].map(cabin_dict)

    # We drop multiple columns that contain a lot of NaN values
    # in this assignment
    # Maybe we should
    X_full.drop(
        ['PassengerId', 'Cabin', 'Name', 'Ticket'],
        inplace=True, axis=1)

    # Instead of dropping the fare column we replace NaN values
    # with the 3rd class passenger fare mean.
    fare_mean = X_full[X_full.Pclass == 3].Fare.mean()
    X_full['Fare'].fillna(fare_mean, inplace=True)
    # Instead of dropping the Embarked column we replace NaN values
    # with `S` denoting Southampton, the most common embarking
    # location
    X_full['Embarked'].fillna('S', inplace=True)
    
    # replace male age nan with median of male age
    # replace femal age nan with median of female age
    male_median_age = X_full[X_full.Sex == "male"].Age.median()
    female_median_age = X_full[X_full.Sex == "female"].Age.median()
    #X_full.Age[(X_full.Sex == "male") & (X_full.Age.isna())] = male_median_age 
    #X_full.Age[(X_full.Sex == "female") & (X_full.Age.isna())] = female_median_age

    X_full.loc[(X_full.Sex == 'male') & (X_full.Age.isna()), 'Age'] = male_median_age 
    X_full.loc[(X_full.Sex == 'female') & (X_full.Age.isna()), 'Age'] = female_median_age 
    

    # We then use the get_dummies function to transform text
    # and non-numerical values into binary categories.
    X_dummies = pd.get_dummies(
        X_full,
        columns=['Sex', 'Cabin_mapped', 'Embarked'],
        drop_first=True)

    # We now have the cleaned data we can use in the assignment
    X = X_dummies[:len(train)]
    submission_X = X_dummies[len(train):]
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)
    return (X_train, y_train), (X_test, y_test), submission_X


def rfc_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a random forest classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    clf = RandomForestClassifier()
    clf.fit(X_train, t_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(t_test, y_pred)
    precision = precision_score(t_test, y_pred)
    recall = recall_score(t_test, y_pred)
    return (accuracy, precision, recall)


def gb_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a Gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    clf = GradientBoostingClassifier()
    clf.fit(X_train, t_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(t_test, y_pred)
    precision = precision_score(t_test, y_pred)
    recall = recall_score(t_test, y_pred)
    return (accuracy, precision, recall)


def param_search(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    gb_param_grid = {
        'n_estimators': list(range(1,100,1)),
        'max_depth': list(range(1,50,1)),
        'learning_rate': list(np.linspace(0.001, 1, 1000))}
    # Instantiate the regressor
    gb = GradientBoostingClassifier()
    # Perform random search
    gb_random = RandomizedSearchCV(
        param_distributions=gb_param_grid,
        estimator=gb,
        scoring="accuracy",
        verbose=0,
        n_iter=50,
        cv=4)
    # Fit randomized_mse to the data
    gb_random.fit(X, y)
    # Print the best parameters and lowest RMSE
    print("Best parameters found: ", gb_random.best_params_)
    print("Lowest RMSE found: ", np.sqrt(np.abs(gb_random.best_score_)))
    return gb_random.best_params_


def gb_optimized_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test) with
    your own optimized parameters
    '''
    clf = GradientBoostingClassifier(n_estimators=	4, max_depth = 5, learning_rate=0.524)
    clf.fit(X_train, t_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(t_test, y_pred)
    precision = precision_score(t_test, y_pred)
    recall = recall_score(t_test, y_pred)
    return (accuracy, precision, recall)


def _create_submission():
    '''Create your kaggle submission
    '''
    # create kaggle submission
    (X_train, y_train), (X_test, y_test), submission_X = get_better_titanic()
    clf = GradientBoostingClassifier(n_estimators=	4, max_depth = 5, learning_rate=0.524)
    clf.fit(X_train, y_train)
    prediction = clf.predict(submission_X)
    build_kaggle_submission(prediction)
    
    
    
################# INDEPENDENT SECTION #################
    
def param_search_indep(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    from scipy.stats import uniform
    learning_rate_distribution = uniform(loc=0, scale=1)
    # Create the parameter grid
    gb_param_grid = {
        'n_estimators': list(range(1,1000,1)),
        'max_depth': list(range(1,100,1)),
        'learning_rate': learning_rate_distribution,
        'max_features': list(range(1,16,1))}
    # Instantiate the regressor
    gb = GradientBoostingClassifier()
    # Perform random search
    gb_random = RandomizedSearchCV(
        param_distributions=gb_param_grid,
        estimator=gb,
        scoring="accuracy",
        verbose=0,
        n_iter=1000,
        cv=None)
    # Fit randomized_mse to the data
    gb_random.fit(X, y)
    # Print the best parameters and lowest RMSE
    print("Best parameters found: ", gb_random.best_params_)
    print("Lowest RMSE found: ", np.sqrt(np.abs(gb_random.best_score_)))
    return gb_random.best_params_

def gp_opt_indep(X_train, t_train, X_test, t_test):
    '''
    Train a gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test) with
    your own optimized parameters
    '''
    #clf = GradientBoostingClassifier(n_estimators=	74, max_features=6 ,max_depth = 5, learning_rate=0.07884937591583496)
    clf = GradientBoostingClassifier(n_estimators=	906, max_features=11 ,max_depth = 2, learning_rate=0.017316576051416455)
    clf.fit(X_train, t_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(t_test, y_pred)
    precision = precision_score(t_test, y_pred)
    recall = recall_score(t_test, y_pred)
    return (accuracy, precision, recall)
    
# main function
if __name__ == '__main__':
    (tr_X, tr_y), (tst_X, tst_y), submission_X = get_better_titanic()
    
    # section 2.2!!!!!!!
    #print(rfc_train_test(tr_X, tr_y, tst_X, tst_y))

    # list with values from 1 to 50
    
    # section 2.4!!!!!!!
    #print(gb_train_test(tr_X, tr_y, tst_X, tst_y))

    # section 2.5!!!!!!!
    #
    #print(param_search(tr_X, tr_y))
    #print(list(np.linspace(0.01, 1, 100)))

    # section 2.6!!!!!!!
    #print(gb_optimized_train_test(tr_X, tr_y, tst_X, tst_y))
    
    #_create_submission()
    
    #param_search_grid(tr_X, tr_y)
    #print(gp_opt_indep(tr_X, tr_y, tst_X, tst_y))


    # list from 0.01 to 1
    #t = np.linspace(0.01, 1, 100)

    # plot age distribution of tr_X
    #plt.hist(tr_X.Age, bins=30)
    #plt.show()
    
    # Load in the raw data
    # check if data directory exists for Mimir submissions
    # DO NOT REMOVE
    """ if os.path.exists('./data/train.csv'):
        train = pd.read_csv('./data/train.csv')
        test = pd.read_csv('./data/test.csv')
    else:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')

    # Concatenate the train and test set into a single dataframe
    # we drop the `Survived` column from the train set
    X_full = pd.concat([train.drop('Survived', axis=1), test], axis=0)
    
    ff = X_full[:30]
    print(ff)
    ff.loc[(ff['Sex'] == 'male') & (ff['Age'].isna()), 'Age'] = 100
    print(ff) """

    # independent section
    
    """ from scipy.stats import uniform
    distt = uniform(loc=0, scale=1)
    # sample from the distribution
    # and make a histogram
    samp = distt.rvs(1000)
    print(samp)
    plt.hist(samp, bins=100)
    plt.show() """
    #print(param_search_indep(tr_X, tr_y))
    print(gp_opt_indep(tr_X, tr_y, tst_X, tst_y))