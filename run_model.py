# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 11:29:23 2016

@author: Elvan
"""


import pandas as pd

print("Load the data using pandas")
#train = pd.read_csv("train_numeric_onehot_all.csv")
#test = pd.read_csv("test_numeric_onehot_all.csv")
train = pd.read_csv("train_numeric_likelihood.csv")
test = pd.read_csv("test_numeric_likelihood.csv")

print("Size of the train data")
print(train.shape)
test['target'] = -1
all_data = train.append(test)

# split train and test
train = all_data[all_data['target']>=0].copy()
test = all_data[all_data['target']<0].copy()

response, trainID = train['target'], train['ID']
train = train.drop(['ID','target'], axis = 1)

testID = test['ID']
response_test = test['target']
test = test.drop(['ID','target'], axis = 1)

from StackingFW import Stacking
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn import cross_validation
import xgboost as xgb
from sklearn.linear_model import SGDClassifier
from sklearn.mixture import GMM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

n_trees = 200



# Generate a list of base (level 0) classifiers.
clfs = [xgb.XGBClassifier(learning_rate = 0.1, min_child_weight = 3, subsample = 0.8, colsample_bytree = 0.8),
        RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, criterion='gini'),
        ExtraTreesClassifier(n_estimators=n_trees, n_jobs=-1, criterion='gini'),
        LogisticRegression(solver = 'liblinear', penalty = 'l1', n_jobs = -1),
        LogisticRegression(solver = 'liblinear', penalty = 'l1', C = .01, n_jobs = -1),    
        LogisticRegression(solver = 'liblinear', penalty = 'l1', C = 100, n_jobs = -1),    
        LogisticRegression(solver = 'liblinear', penalty = 'l2', n_jobs = -1),   
        LogisticRegression(solver = 'liblinear', penalty = 'l2', C = .01, n_jobs = -1),    
        LogisticRegression(solver = 'liblinear', penalty = 'l2', C = 100, n_jobs = -1),     
        KNeighborsClassifier(n_neighbors=25, p = 1, weights = 'distance'),
        KNeighborsClassifier(n_neighbors=25, p = 2, weights = 'distance'),
        KNeighborsClassifier(n_neighbors=25, metric = 'chebyshev', weights = 'distance'),
        GradientBoostingClassifier(learning_rate = 0.1), # iyi ama yavas
        SGDClassifier(loss="log", penalty="none", n_jobs = -1, eta0 = 1e-10, learning_rate = 'constant', n_iter = 5),
        SGDClassifier(loss="log", penalty="l2",   n_jobs = -1, eta0 = 1e-10, learning_rate = 'constant', n_iter = 5, alpha = 0.0001),
        SGDClassifier(loss="log", penalty="l2",   n_jobs = -1, eta0 = 1e-10, learning_rate = 'constant', n_iter = 5, alpha = 0.01),
        SGDClassifier(loss="log", penalty="l2",   n_jobs = -1, eta0 = 1e-10, learning_rate = 'constant', n_iter = 5, alpha = 1),
        SGDClassifier(loss="log", penalty="elasticnet",  n_jobs = -1,  eta0 = 1e-10, learning_rate = 'constant', n_iter = 5, alpha = 0.0001),    
        SGDClassifier(loss="log", penalty="elasticnet",   n_jobs = -1,  eta0 = 1e-10,    learning_rate = 'constant',     n_iter = 5,  alpha = 0.01),    
        SGDClassifier(loss="log",   penalty="elasticnet",   n_jobs = -1,   eta0 = 1e-10, learning_rate = 'constant',    n_iter = 5,  alpha = 1),        
        SGDClassifier(loss="log",   penalty="l1",   n_jobs = -1,   eta0 = 1e-10,  learning_rate = 'constant',   n_iter = 5,   alpha = 0.0001),
        SGDClassifier(loss="log",   penalty="l1",   n_jobs = -1,   eta0 = 1e-10,   learning_rate = 'constant',  n_iter = 5,  alpha = 0.01),
        SGDClassifier(loss="log",    penalty="l1",   n_jobs = -1,   eta0 = 1e-10,   learning_rate = 'constant',  n_iter = 5,  alpha = 1),
        LinearDiscriminantAnalysis(),
        #QuadraticDiscriminantAnalysis(reg_param = 0.2),
        #AdaBoostClassifier(n_estimators = n_trees) # orta halli ama cok yavas
        #SVC(C = 2.0, probability = True), # Asiri yavas, burada donuyor
        #DecisionTreeClassifier(criterion = 'gini'), # Just not working!!
        #GaussianNB(),
        #BernoulliNB(),
        #BaggingClassifier(),
        #GMM(n_components = 2),# Just not working!!
        ]
        

# The training sets will be used for all training and validation purposes.
# The testing sets will only be used for evaluating the final blended (level 1) classifier.
X_train, y_train = train, response

n_folds = 10

#TODO Automatize the stacking process by adding more layers

# Generate k stratified folds of the training data.
skf = list(cross_validation.StratifiedKFold(y_train, n_folds, shuffle=True))
stk = Stacking(xgb.XGBClassifier, clfs, skf, stackingc=False, proba=True)
stk.fit(X_train, y_train)

probs = stk.predict_proba(test)

final_test_preds = probs[:,1]

preds_out = pd.DataFrame({"ID": testID.astype(int), "PredictedProb": final_test_preds})
preds_out = preds_out.set_index('ID')
preds_out.to_csv('xgb_submission2.csv')