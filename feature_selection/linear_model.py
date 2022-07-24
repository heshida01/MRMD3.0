"""
Lasso and Elasticnet are a regression problem, not implemented in skLearn
So we added a logistic regression on the basis of them.

"""
from sklearn.linear_model import *
import pandas as pd
import numpy as np
from  sklearn.linear_model import RidgeClassifier,LogisticRegression,ElasticNet
from sklearn.preprocessing import MinMaxScaler
import sys


def lasso(file,mode = "None"):
    dataset = pd.read_csv(file,engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)

    X = dataset[:, 1:]
    y = dataset[:, 0]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    if mode == "Lasso":
        lasso = Lasso()
        lasso.fit(X, y)
        result = [(x, y) for x, y in zip(features_name[1:], lasso.coef_)]
        result = sorted(result, key=lambda x: abs(x[1]), reverse=True)
    elif mode == "Ridge":
        ridge = RidgeClassifier()
        ridge.fit(X, y)
        result = [(x, y) for x, y in zip(features_name[1:], ridge.coef_)]
        result = sorted(result, key=lambda x: abs(x[1]), reverse=True)
    elif mode == "ElasticNet":
        elasticnet = ElasticNet()
        elasticnet.fit(X, y)
        result = [(x, y) for x, y in zip(features_name[1:], elasticnet.coef_)]
        result = sorted(result, key=lambda x: abs(x[1]), reverse=True)
    if mode != "None":
        return [x[0] for x in result]

    lasso = Lasso()
    lasso.fit(X, y)
    result = [(x, y) for x, y in zip(features_name[1:], lasso.coef_)]
    result1 = sorted(result, key=lambda x: abs(x[1]), reverse=True)


    ridge = RidgeClassifier()
    ridge.fit(X, y)
    result = [(x, y) for x, y in zip(features_name[1:], ridge.coef_)]
    result2 = sorted(result, key=lambda x: abs(x[1]), reverse=True)


    elasticnet = ElasticNet()
    elasticnet.fit(X, y)

    result = [(x, y) for x, y in zip(features_name[1:], elasticnet.coef_)]
    result3 = sorted(result, key=lambda x: abs(x[1]), reverse=True)

    return ([x[0] for x in result1 ],
            [x[0] for x in result2 ],
            [x[0] for x in result3 ])





def linear_models(file,mode='Lasso'):
    dataset = pd.read_csv(file,engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)

    X = dataset[:, 1:]
    y = dataset[:, 0]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    if mode == "lasso":
        lasso = Lasso()
        lasso.fit(X, y)
        result = [(x, y) for x, y in zip(features_name[1:], lasso.coef_)]
        result = sorted(result, key=lambda x: abs(x[1]), reverse=True)
    elif mode == "ridge":
        ridge = RidgeClassifier()
        ridge.fit(X, y)
        result = [(x, y) for x, y in zip(features_name[1:], ridge.coef_)]
        result = sorted(result, key=lambda x: abs(x[1]), reverse=True)
    elif mode == "elasticnet":
        elasticnet = ElasticNet()
        elasticnet.fit(X, y)
        result = [(x, y) for x, y in zip(features_name[1:], elasticnet.coef_)]
        result = sorted(result, key=lambda x: abs(x[1]), reverse=True)
    if mode != "None":
        return [x[0] for x in result]


def run(csvfile,logger):
    logger.info('linear model start...')
    feature_list = lasso(csvfile)

    logger.info('linear model end.')
    return feature_list
#

if __name__ == '__main__':

    print(linear_models(sys.argv[1],sys.argv[2]))
