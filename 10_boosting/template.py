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
    
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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

    fare_mean = X_full[X_full.Pclass == 3].Fare.mean()
    X_full['Fare'].fillna(fare_mean, inplace=True)

    X_full['Embarked'].fillna('S', inplace=True)
    
    male_median_age = X_full[X_full.Sex == "male"].Age.median()
    female_median_age = X_full[X_full.Sex == "female"].Age.median()
    X_full.loc[(X_full.Sex == 'male') & (X_full.Age.isna()), 'Age'] = male_median_age 
    X_full.loc[(X_full.Sex == 'female') & (X_full.Age.isna()), 'Age'] = female_median_age 
    
    # do one hot encoding on pclass
    pclass_dummies = pd.get_dummies(X_full.Pclass, prefix='Pclass')
    X_full = pd.concat([X_full, pclass_dummies], axis=1)
    X_full.drop('Pclass', axis=1, inplace=True)
    
    # do one hot encoding on Parch and SibSp
    parch_dummies = pd.get_dummies(X_full.Parch, prefix='Parch')
    X_full = pd.concat([X_full, parch_dummies], axis=1)
    X_full.drop('Parch', axis=1, inplace=True)
    
    sibsp_dummies = pd.get_dummies(X_full.SibSp, prefix='SibSp')
    X_full = pd.concat([X_full, sibsp_dummies], axis=1)
    X_full.drop('SibSp', axis=1, inplace=True)
    
    # one hot encoding for age
    # split age int children, young, middle, old
    X_full.loc[X_full.Age < 7, 'Age'] = 0
    X_full.loc[(X_full.Age >= 7) & (X_full.Age < 20), 'Age'] = 1
    X_full.loc[(X_full.Age >= 20) & (X_full.Age < 55), 'Age'] = 2
    X_full.loc[(X_full.Age >= 55), 'Age'] = 3
    
    age_dummies = pd.get_dummies(X_full.Age, prefix='Age')
    X_full = pd.concat([X_full, age_dummies], axis=1)
    X_full.drop('Age', axis=1, inplace=True)
    
    # one hot encoding for fare
    # split fare into low, medium, high
    X_full.loc[X_full.Fare < 7, 'Fare'] = 0
    X_full.loc[(X_full.Fare >= 7) & (X_full.Fare < 20), 'Fare'] = 1
    X_full.loc[(X_full.Fare >= 20), 'Fare'] = 2
    
    fare_dummies = pd.get_dummies(X_full.Fare, prefix='Fare')
    X_full = pd.concat([X_full, fare_dummies], axis=1)
    X_full.drop('Fare', axis=1, inplace=True)
    
    X_dummies = pd.get_dummies(
        X_full,
        columns=['Sex', 'Cabin_mapped', 'Embarked'],
        drop_first=True)

    X = X_dummies[:len(train)]
    submission_X = X_dummies[len(train):]
    
    y = train.Survived
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=5, stratify=y)
    """ X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=5, stratify=y) """
    #return (np.array(X), np.array(y)), np.array(submission_X)
    # return (X_train, y_train), (X_test, y_test), submission_X as np.array
    return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test)), np.array(submission_X)

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()

       
        self.fc_1 = nn.Linear(36, 50)
        self.fc_2 = nn.Linear(50, 50)
        self.fc_3 = nn.Linear(50, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        This method performs the forward pass,
        x is the input feature being passed through
        the network at the current time
        '''
        x = self.fc_1(x)
        x = self.sigmoid(x)
        x = self.fc_2(x)
        x = self.sigmoid(x)
        x = self.fc_3(x)
        x = self.sigmoid(x)

        return x

class TitanicDataset(Dataset):
    def __init__(self):
        '''
        A simple PyTorch dataset for the Iris data
        '''

        #features, targets, self.classes = load_iris()
        (X, y), (X_test, y_test), submission_X = indep_get_titanic()
        #(X, y), submission_X = indep_get_titanic()
        # we first have to convert the numpy data to compatible
        # PyTorch data:
        # * Features should be of type float
        # * Class labels should be of type long
        
        self.features = torch.from_numpy(X).float()
        self.targets = torch.from_numpy(y).long()
        self.classes = np.unique(y)

    def __len__(self):
        '''We always have to define this method
        so PyTorch knows how many items are in our dataset
        '''
        return self.features.shape[0]

    def __getitem__(self, i):
        '''We also have to define this method to tell
        PyTorch what the i-th element should be. In our
        case it's simply the i-th elements from both features
        and targets
        '''
        return self.features[i, :], self.targets[i]

def create_titanic_data_loader():
    '''Another convinient thing in PyTorch is the dataloader
    It allows us to easily iterate over all the data in our
    dataset. We can also:
    * set a batch size. In short, setting a batch size > 1
    allows us to train on more than 1 sample at a time and this
    generally decreases training time
    * shuffe the data.
    '''
    dl = DataLoader(TitanicDataset(), batch_size=10, shuffle=True)
    return dl

def train_simple_model():
    # Set up the data
    ds = TitanicDataset()
    dl = DataLoader(ds, batch_size=10, shuffle=True)

    # Initialize the model
    model = SimpleNetwork()

    # Choose a loss metric, cross entropy is often used
    #loss_metric = nn.CrossEntropyLoss()
    loss_metric = nn.BCELoss()
    # Choose an optimizer and connect to model weights
    # Often the Adam optimizer is used
    #optimizer = torch.optim.SGD(model.parameters(),lr=0.002)
    optimizer = torch.optim.Adam(model.parameters())

    num_steps = 0
    loss_values = []
    # THE TRAINING LOOP
    # we will do 50 epochs (i.e. we will train on all data 50 times)
    for epoch in range(2):
        for (feature, target) in dl:
            feature = feature.type(torch.FloatTensor)
            target = target.type(torch.LongTensor)
            num_steps += 1

            optimizer.zero_grad()

            out = model(feature)
            
            loss = loss_metric(out, target.unsqueeze(1).float())
            # To perform the backward propagation we do:
            loss.backward()
            # The optimizer then tunes the weights of the model
            optimizer.step()

            #if num_steps % 200 == 0:
                #print("the loss is currently: ", loss.item())
                #loss_values.append(loss.mean().item())

    #plt.plot(loss_values)
    #plt.title('Loss as a function of training steps')
    #plt.show()
    # return the trained model
    return model


def _create_submission_indep():
    '''Create your kaggle submission
    '''
    # create kaggle submission
    (X, y), submission_X = indep_get_titanic()
    trained_model = train_simple_model()
    submission_X = torch.tensor(submission_X)
    submission_X = submission_X.type(torch.FloatTensor)
    outputs = trained_model(submission_X)
    outputs = outputs.detach().numpy()
    # outputs is from sigmoid, round for 0 or 1
    outputs = np.round(outputs)
    # make outputs a int
    outputs = np.squeeze(outputs)
    # make all elements in outputs integers
    prediction = outputs.astype(int)

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
    
    """ from scipy.stats import uniform
    distt = uniform(loc=0, scale=1)
    # sample from the distribution
    # and make a histogram
    samp = distt.rvs(1000)
    print(samp)
    plt.hist(samp, bins=100)
    plt.show() """
    #print(param_search_indep(tr_X, tr_y))
    #print(gp_opt_indep(tr_X, tr_y, tst_X, tst_y))
    
    (X, y), (X_test, y_test), submission_X = indep_get_titanic()

    #(tr_X, tr_y), (tst_X, tst_y), submission_X = indep_get_titanic()
    # list for trained models
    num_learners = 10
    trained_model_list = []
    for i in range(num_learners):
        trained_model_list.append(train_simple_model())
    
    
    # predict on test set
    # with each model
    # pick prediciton with most votes
    X_test = torch.tensor(X_test)
    X_test = X_test.type(torch.FloatTensor)
    predictions = np.zeros((len(y_test), num_learners))
    k = 0
    for model in trained_model_list:
        output = model(X_test)
        #print(output)
        output = output.detach().numpy()
        output = np.round(output)
        output = np.squeeze(output)
        predictions[:, k] = output
        k += 1
    
    # conver to int
    predictions = predictions.astype(int)
    
    print(predictions[:20, 0])
    print(predictions[:20, 1])
    print(predictions[:20, 2])
    
    # majority vote
    #print(predictions[:, 1])
    final_prediction = np.zeros(len(y_test))
    for i in range(len(y_test)):
        final_prediction[i] = np.argmax(np.bincount(predictions[i]))
    
    print(final_prediction)
    accuracy = np.sum(final_prediction== y_test) / len(y_test)
    precision = precision_score(y_test, final_prediction)
    recall = recall_score(y_test, final_prediction)
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall)
    
    #trained_model = train_simple_model()
    
    # test the model, compute accuracy
    # put y into trained model and compare to y
    # make y a tensor
    
    """ X_test = torch.tensor(X_test)
    X_test = X_test.type(torch.FloatTensor)
    outputs = trained_model(X_test)
    outputs = outputs.detach().numpy()
    # outputs is from sigmoid, round for 0 or 1
    # if output is 0.6 or higher, round to 1
    outputs = np.round(outputs)
    # make outputs a int
    outputs = np.squeeze(outputs)
    # given that the output is from a sigmoid function
    # we can just round the output to get the prediction
    # and then calculate the accuracy
     """
    
    """ # compute precision and recall
    accuracy = np.sum(outputs== y_test) / len(y_test)
    precision = precision_score(y_test, outputs)
    recall = recall_score(y_test, outputs)
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("recall: ", recall) """

    
    
    #_create_submission_indep()