from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
import pandas as pd
import numpy as np
from multiprocessing import Pool, Queue,Process,Manager
import  sys
tree_models = ['DecisionTree','RandomForest','GradientBoosting','ExtraTreesClassifier']
def fun1(X,y,mydict):
    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(X, y)
    lis1 = rf.feature_importances_
    mydict['fun1'] = lis1
    return lis1


def fun2(X,y,mydict):
    dt = DecisionTreeClassifier()
    dt.fit(X, y)
    lis2 = dt.feature_importances_
    mydict['fun2'] = lis2
    return lis2


def fun3(X,y,mydict):
    gb = GradientBoostingClassifier()
    gb.fit(X, y)
    lis3 = gb.feature_importances_
    mydict['fun3'] = lis3
    return lis3

def fun4(X,y,mydict):
    et = ExtraTreesClassifier()
    et.fit(X, y)
    lis4= et.feature_importances_
    mydict['fun4'] = lis4
    return lis4

def tree_Fimportance(file,model = "None"):
    dataset= pd.read_csv(file)

    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)
    X = dataset[:, 1:]
    y = dataset[:, 0]

    if model != "None":
        if model == "DecisionTree":
            m = DecisionTreeClassifier()
            m.fit(X,y)
            lis1 = m.feature_importances_
            res = [(x, y) for x, y in zip(features_name[1:], lis1)]
            return [x[0] for x in res]
        elif model == "RandomForest":
            m = RandomForestClassifier()
            m.fit(X,y)
            lis1 = m.feature_importances_
            res = [(x, y) for x, y in zip(features_name[1:], lis1)]
            return [x[0] for x in res]
        elif model == "GradientBoosting":
            m = GradientBoostingClassifier()
            m.fit(X,y)
            lis1 = m.feature_importances_
            res = [(x, y) for x, y in zip(features_name[1:], lis1)]
            return [x[0] for x in res]
        elif model == "ExtraTreesClassifier":
            m = ExtraTreesClassifier()
            m.fit(X,y)
            lis1 = m.feature_importances_
            res = [(x, y) for x, y in zip(features_name[1:], lis1)]
            return [x[0] for x in res]

    mydict = Manager().dict()


    task_1 = Process(target=fun1, args=(X,y,mydict))
    task_2 = Process(target=fun2, args=(X,y,mydict))
    task_3 = Process(target=fun3, args=(X,y,mydict))
    task_4 = Process(target=fun4, args=(X,y,mydict))


    task_1.start()
    task_2.start()
    task_3.start()
    task_4.start()


    task_1.join()
    task_2.join()
    task_3.join()
    task_4.join()

    lis1 = mydict['fun1']
    lis2 = mydict['fun2']
    lis3 = mydict['fun3']
    lis4 = mydict['fun4']




    result1 = [(x, y) for x, y in zip(features_name[1:], lis1)]
    result1 = sorted(result1, key=lambda x: x[1], reverse=True)
    result2 = [(x, y) for x, y in zip(features_name[1:], lis2)]
    result2 = sorted(result2, key=lambda x: x[1], reverse=True)
    result3 = [(x, y) for x, y in zip(features_name[1:], lis3)]
    result3 = sorted(result3, key=lambda x: x[1], reverse=True)
    result4 = [(x, y) for x, y in zip(features_name[1:], lis4)]
    result4 = sorted(result4, key=lambda x: x[1], reverse=True)

    return ([x[0] for x in result1],
            [x[0] for x in result2],
            [x[0] for x in result3],
            [x[0] for x in result4], )




def run(csvfile, logger):
    logger.info('tree feature importance start...')
    feature_list = tree_Fimportance(csvfile)
    logger.info('tree feature importance  end.')
    return feature_list


def tree_models(inputfile,mode):
    dataset= pd.read_csv(inputfile)
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)
    X = dataset[:, 1:]
    y = dataset[:, 0]

    if mode == "1":
        rf = RandomForestClassifier(n_jobs=-1)
        rf.fit(X, y)
        result = rf.feature_importances_

    elif mode == "2":
        dt = DecisionTreeClassifier()
        dt.fit(X, y)
        result = dt.feature_importances_

    elif mode == "3":
        gb = GradientBoostingClassifier()
        gb.fit(X, y)
        result = gb.feature_importances_

    elif mode == "4":
        et = ExtraTreesClassifier()
        et.fit(X, y)
        result = et.feature_importances_

    result1 = [(x, y) for x, y in zip(features_name[1:], result)]
    result1 = sorted(result1, key=lambda x: x[1], reverse=True)

    return [x[0] for x in result1]

if __name__ == '__main__':
    csvfile,model = sys.argv[1],sys.argv[2]
    print(tree_Fimportance(csvfile,model))

