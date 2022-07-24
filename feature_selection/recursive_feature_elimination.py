from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB,MultinomialNB ,BernoulliNB
from sklearn.tree import *
from sklearn.ensemble import *
import time
import sys

# rfe_base_models = ['LinearSVC',"LinearSVR",
#            'ComplementNB','MultinomialNB','BernoulliNB',
#             "Lasso","Ridge","ElasticNet",
#             "DecisionTreeClassifier","RandomForestClassifier","GradientBoostingClassifier"
#             ]

def ref_(file,mode = "None"):

    dataset = pd.read_csv(file,engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)
    X = dataset[:, 1:]
    y = dataset[:, 0]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    if mode == "LogisticRegression" or mode == "1":
        estimator = LogisticRegression()
        selector = RFE(estimator=estimator, n_features_to_select=5)
        selector.fit_transform(X, y)
        result = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))
    elif mode == "LinearSVC" or  mode == "2":
        estimator = LinearSVC()
        selector = RFE(estimator=estimator, n_features_to_select=5)
        selector.fit_transform(X, y)
        result = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))
    elif mode == "DecisionTreeClassifier" or mode == "3":
        estimator = DecisionTreeClassifier()
        selector = RFE(estimator=estimator, n_features_to_select=5)
        selector.fit_transform(X, y)
        result = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))

    if mode != "None":
        return [x[1] for x in result]

    estimator = LogisticRegression()
    selector = RFE(estimator=estimator, n_features_to_select=5)
    selector.fit_transform(X, y)
    result1 = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))

    estimator = LinearSVC()
    selector = RFE(estimator=estimator, n_features_to_select=5)
    selector.fit_transform(X, y)
    result2 = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))

    estimator = DecisionTreeClassifier()
    selector = RFE(estimator=estimator, n_features_to_select=5)
    selector.fit_transform(X, y)
    result3 = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))


    return ([x[1] for x in result1],
            [x[1] for x in result2],
            [x[1] for x in result3])



# def ref_svm(file):
#     dataset = pd.read_csv(file,engine='python').dropna(axis=1)
#     features_name = dataset.columns.values.tolist()
#     dataset = np.array(dataset)
#     X = dataset[:, 1:]
#     y = dataset[:, 0]
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)
#
#     estimator = LinearSVC()
#     selector = RFE(estimator=estimator, n_features_to_select=5)
#     selector.fit_transform(X, y)
#     result = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))
#
#     return [x[1] for x in result]




# def ref_model(file,model):
#     dataset = pd.read_csv(file, engine='python').dropna(axis=1)
#     features_name = dataset.columns.values.tolist()
#     dataset = np.array(dataset)
#     X = dataset[:, 1:]
#     y = dataset[:, 0]
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)
#
#     estimator =eval(model)()
#     selector = RFE(estimator=estimator, n_features_to_select=1)
#     selector.fit_transform(X, y)
#     result = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))
#
#     return [x[1] for x in result]
#
# def ref_model_test(file,model=None):
#     dataset = pd.read_csv(file, engine='python').dropna(axis=1)
#     features_name = dataset.columns.values.tolist()
#     dataset = np.array(dataset)
#     X = dataset[:, 1:]
#     y = dataset[:, 0]
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)
#
#     models = rfe_base_models
#     for model in models:
#         print(model,end = '')
#         estimator =eval(model)()
#
#         estimator.fit(X,y)
#         selector = RFE(estimator=estimator, n_features_to_select=1)
#         selector.fit_transform(X, y)
#         result = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))
#         print(result)
#
#     return [x[1] for x in result]

def run(csvfile,logger):
    logger.info('ref start...')
    feature_list = ref_(csvfile)
    logger.info('ref end.')
    return feature_list

if __name__ == '__main__':
    file,mode = sys.argv[1],sys.argv[2]
    res = ref_(file,mode)
    print(res)
