from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import *
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import *
import sys
from multiprocessing import  Process,Queue,Manager


def task(model, X, y, features_name, d):
    estimator =model
    selector = RFE(estimator=estimator, n_features_to_select=5)
    selector.fit_transform(X, y)
    result1 = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))
    d[str(model)] = [x[1] for x in result1]
    return  result1


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

    models = [LogisticRegression(),LinearSVC(),SGDClassifier()]
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


def run(csvfile,logger):
    logger.info('ref start...')
    feature_list = ref_(csvfile)
    logger.info('ref end.')
    return feature_list

if __name__ == '__main__':
    file,mode = sys.argv[1],sys.argv[2]
    res = ref_(file,mode)
    print(res)
