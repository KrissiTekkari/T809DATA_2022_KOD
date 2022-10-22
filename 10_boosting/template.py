# Author: Kristjan Orri Dadason
# Date: 15/10/2022
# Project: Boosting
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.


import os
from black import out
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    # replace female age nan with median of female age
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
    
#/////////////////////////////////////////////////////#    
#######################################################
################# INDEPENDENT SECTION #################
#######################################################
#/////////////////////////////////////////////////////#

def indep_get_titanic():

    if os.path.exists('./data/train.csv'):
        train = pd.read_csv('./data/train.csv')
        test = pd.read_csv('./data/test.csv')
    else:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')

    X_full = pd.concat([train.drop('Survived', axis=1), test], axis=0)


    X_full['Cabin_mapped'] = X_full['Cabin'].astype(str).str[0]

    cabin_dict = {k: i for i, k in enumerate(X_full.Cabin_mapped.unique())}
    X_full.loc[:, 'Cabin_mapped'] =\
        X_full.loc[:, 'Cabin_mapped'].map(cabin_dict)

    X_full.drop(
        ['PassengerId', 'Cabin', 'Name', 'Ticket'],
        inplace=True, axis=1)

    fare_median = X_full[X_full.Pclass == 3].Fare.median()
    X_full['Fare'].fillna(fare_median, inplace=True)

    X_full['Embarked'].fillna('S', inplace=True)
    
    """ male_median_age = X_full[X_full.Sex == "male"].Age.median()
    female_median_age = X_full[X_full.Sex == "female"].Age.median()
    X_full.loc[(X_full.Sex == 'male') & (X_full.Age.isna()), 'Age'] = male_median_age 
    X_full.loc[(X_full.Sex == 'female') & (X_full.Age.isna()), 'Age'] = female_median_age """
    
    mean_age = X_full.Age.mean()
    std_age = X_full.Age.std()
    random_age = np.random.randint(mean_age - std_age, mean_age + std_age, size=X_full.Age.isna().sum())
    X_full.loc[X_full.Age.isna(), 'Age'] = random_age
    
    """ # do one hot encoding on pclass
    pclass_dummies = pd.get_dummies(X_full.Pclass, prefix='Pclass')
    X_full = pd.concat([X_full, pclass_dummies], axis=1)
    X_full.drop('Pclass', axis=1, inplace=True)
    
    # do one hot encoding on Parch and SibSp
    parch_dummies = pd.get_dummies(X_full.Parch, prefix='Parch')
    X_full = pd.concat([X_full, parch_dummies], axis=1)
    X_full.drop('Parch', axis=1, inplace=True)
    
    sibsp_dummies = pd.get_dummies(X_full.SibSp, prefix='SibSp')
    X_full = pd.concat([X_full, sibsp_dummies], axis=1)
    X_full.drop('SibSp', axis=1, inplace=True) """

    """ # one hot encoding for age
    # split age int children, young, middle, old
    X_full.loc[X_full.Age < 18, 'Age'] = 0
    X_full.loc[(X_full.Age >= 18) & (X_full.Age < 40), 'Age'] = 1
    X_full.loc[(X_full.Age >= 40), 'Age'] = 2
    
    age_dummies = pd.get_dummies(X_full.Age, prefix='Age')
    X_full = pd.concat([X_full, age_dummies], axis=1)
    X_full.drop('Age', axis=1, inplace=True) """
    
    """ # one hot encoding for fare
    # split fare into below and equal to median, above median
    fare_median = X_full.Fare.median()
    X_full.loc[X_full.Fare <= fare_median, 'Fare'] = 0
    X_full.loc[X_full.Fare > fare_median, 'Fare'] = 1
    
    fare_dummies = pd.get_dummies(X_full.Fare, prefix='Fare')
    X_full = pd.concat([X_full, fare_dummies], axis=1)
    X_full.drop('Fare', axis=1, inplace=True) """
    
    X_dummies = pd.get_dummies(
        X_full,
        columns=['Sex', 'Cabin_mapped', 'Embarked'],
        drop_first=True)

    X = X_dummies[:len(train)]
    submission_X = X_dummies[len(train):]
    
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y)
    return (X_train, y_train), (X_test, y_test), submission_X


def indep_param_search(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    gb_param_grid = {
        'max_depth': list(range(1,50,1)),
        'criterion': ["gini", "entropy"],
        'min_samples_split': list(range(2, 10, 1)),
        'min_samples_leaf': list(range(1, 10, 1)),
        'min_weight_fraction_leaf': list(np.linspace(0, 0.5, 100)),
        'max_features': list(range(1, X.shape[1], 1)),
        'max_leaf_nodes': list(range(2, 100, 1)),
        'class_weight': [None, 'balanced']
        }
    # Instantiate the regressor
    rfc = RandomForestClassifier(n_estimators=100, bootstrap=True, random_state=5)
    # Perform random search
    rfc_random = RandomizedSearchCV(
        param_distributions=gb_param_grid,
        estimator=rfc,
        scoring="accuracy",
        verbose=0,
        n_iter=1000,
        cv=4)
    # Fit randomized_mse to the data
    rfc_random.fit(X, y)
    # Print the best parameters and lowest RMSE
    print("Best parameters found: ", rfc_random.best_params_)
    print("Lowest RMSE found: ", np.sqrt(np.abs(rfc_random.best_score_)))
    return rfc_random.best_params_


def _create_submission_indep():
    '''Create your kaggle submission
    '''
    # create kaggle submission
    (X_train, y_train), (X_test, y_test), submission_X = indep_get_titanic()
    #clf = RandomForestClassifier(n_estimators=100, bootstrap = True)
    clf = RandomForestClassifier(n_estimators=100, min_weight_fraction_leaf=0.0303, 
                                      min_samples_split=3, min_samples_leaf=6, max_leaf_nodes=44,
                                      max_features=11, max_depth=16, criterion='gini', 
                                      class_weight='balanced', bootstrap=True)
    clf.fit(X_train, y_train)
    prediction = clf.predict(submission_X)
    build_kaggle_submission(prediction)

# main function
if __name__ == '__main__':
    #(tr_X, tr_y), (tst_X, tst_y), submission_X = get_better_titanic()
    
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
    (X_train, Y_train), (X_test, Y_test), submission_X = indep_get_titanic()
    # Random Forest

    """ random_forest = RandomForestClassifier(n_estimators=100, bootstrap = True)
    random_forest.fit(X_train, Y_train)

    
    Y_prediction_forest = random_forest.predict(X_test)

    #acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)
    acc_random_forest = round(accuracy_score(Y_test, Y_prediction_forest) * 100, 2)
    
    print(f"rfc accuracy: {acc_random_forest}")

    from sklearn.model_selection import cross_val_score
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_leaf = 1, min_samples_split = 10,bootstrap = True)
    scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std()) """
    
    #_create_submission_indep()
    #print(indep_param_search(X_train, Y_train))
    from sklearn.model_selection import cross_val_score
    rf = RandomForestClassifier(n_estimators=100, min_weight_fraction_leaf=0.0303, min_samples_split=3, min_samples_leaf=6, max_leaf_nodes=44, max_features=11, max_depth=16, criterion='gini', class_weight='balanced', bootstrap=True)
    rf.fit(X_train, Y_train)
    Y_prediction_forest = rf.predict(X_test)
    scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())
    
    acc_random_forest = round(accuracy_score(Y_test, Y_prediction_forest) * 100, 2)
    print(f"rfc accuracy: {acc_random_forest}") 