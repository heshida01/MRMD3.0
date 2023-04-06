from sklearn.linear_model import *
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import sys
from multiprocessing import  Process,Queue,Manager

def linear_models(file,mode = "None"):
    dataset = pd.read_csv(file,engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)

    X = dataset[:, 1:]
    y = dataset[:, 0]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    if mode.lower() == "lasso":
        lasso = LogisticRegression(penalty="l1",solver="saga")
        lasso.fit(X, y)
        result = [(x, y) for x, y in zip(features_name[1:], np.sum(np.absolute(lasso.coef_),axis=0))]
        result = sorted(result, key=lambda x: abs(x[1]), reverse=True)
    elif mode.lower() == "ridge":
        ridge = LogisticRegression(penalty="l2",solver="saga")
        ridge.fit(X, y)
        result = [(x, y) for x, y in zip(features_name[1:], np.sum(np.absolute(ridge.coef_),axis=0))]
        result = sorted(result, key=lambda x: abs(x[1]), reverse=True)
    elif mode.lower() == "elasticnet":
        elasticnet = LogisticRegression(penalty="elasticnet",l1_ratio=0.1,solver="saga")
        elasticnet.fit(X, y)
        result = [(x, y) for x, y in zip(features_name[1:], np.sum(np.absolute(elasticnet.coef_),axis=0))]
        result = sorted(result, key=lambda x: abs(x[1]), reverse=True)
    if mode != "None":
        return [x[0] for x in result]

    lasso =  LogisticRegression(penalty="l1",solver="saga")
    lasso.fit(X, y)
    result = [(x, y) for x, y in zip(features_name[1:], np.sum(np.absolute(lasso.coef_),0))]
    result1 = sorted(result, key=lambda x: abs(x[1]), reverse=True)


    ridge = LogisticRegression(penalty="l2",solver="saga")
    ridge.fit(X, y)
    result = [(x, y) for x, y in zip(features_name[1:], np.sum(np.absolute(ridge.coef_),0))]
    result2 = sorted(result, key=lambda x: abs(x[1]), reverse=True)


    elasticnet = LogisticRegression(penalty="elasticnet",l1_ratio=0.1,solver="saga")
    elasticnet.fit(X, y)

    result = [(x, y) for x, y in zip(features_name[1:], np.sum(np.absolute(elasticnet.coef_),0))]
    result3 = sorted(result, key=lambda x: abs(x[1]), reverse=True)

    return ([x[0] for x in result1 ],
            [x[0] for x in result2 ],
            [x[0] for x in result3 ])

def linear_model(file,mode='Lasso'):
    dataset = pd.read_csv(file,engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)

    X = dataset[:, 1:]
    y = dataset[:, 0]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    if mode.lower() == "lasso":
        lasso =LogisticRegression(penalty="l1",solver="saga")
        lasso.fit(X, y)
        result = [(x, y) for x, y in zip(features_name[1:], np.sum(np.absolute(lasso.coef_),0))]
        result = sorted(result, key=lambda x: abs(x[1]), reverse=True)
    elif mode.lower() == "ridge":
        ridge = LogisticRegression(penalty="l2",solver="saga")
        ridge.fit(X, y)
        result = [(x, y) for x, y in zip(features_name[1:], np.sum(np.absolute(ridge.coef_),0))]
        result = sorted(result, key=lambda x: abs(x[1]), reverse=True)
    elif mode.lower() == "elasticnet":
        elasticnet = LogisticRegression(penalty="elasticnet",l1_ratio=0.1,solver="saga")
        elasticnet.fit(X, y)
        result = [(x, y) for x, y in zip(features_name[1:], np.sum(np.absolute(elasticnet.coef_),0))]
        result = sorted(result, key=lambda x: abs(x[1]), reverse=True)

    return [x[0] for x in result]

###
def task(model,X,y,features_name,d):

        model.fit(X, y)

        result = [(x, y) for x, y in zip(features_name[1:], np.sum(np.absolute(model.coef_),0))]

        result = sorted(result, key=lambda x: abs(x[1]), reverse=True)

        d[str(model)] = [x[0] for x in result]


def parallel(file):
    dataset = pd.read_csv(file,engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)

    X = dataset[:, 1:]
    y = dataset[:, 0]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    models = [LogisticRegression(penalty="l1",solver="saga"),LogisticRegression(penalty="l2",solver="saga"), LogisticRegression(penalty="elasticnet",l1_ratio=0.1,solver="saga")]
    with Manager() as manager:
        d = manager.dict()
        tasks = []
        for model in models:
            p = Process(target = task,args=(model,X,y,features_name,d))
            p.start()
            tasks.append(p)
        for t in tasks:
            t.join()

        return d[str(models[0])],d[str(models[1])],d[str(models[2])]



def run(csvfile,logger,mode=""):
    logger.info('linear model start...')
    feature_list = parallel(csvfile)

    #feature_list = linear_models(csvfile)
    logger.info('linear model end.')
    return feature_list



if __name__ == '__main__':
    print(linear_models("../test.csv","lasso"))
    linear_models(sys.argv[1], sys.argv[2])
    print(linear_models(sys.argv[1],sys.argv[2]))

