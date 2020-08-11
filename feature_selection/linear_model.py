from sklearn.linear_model import Lasso,LogisticRegression,Ridge
import pandas as pd
import numpy as np
def lasso(file):
    dataset = pd.read_csv(file,engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)

    X = dataset[:, 1:]
    y = dataset[:, 0]

    lasso = Lasso()
    lasso.fit(X, y)
    result = [(x, y) for x, y in zip(features_name[1:], lasso.coef_)]
    result1 = sorted(result, key=lambda x: abs(x[1]), reverse=True)


    ridge = Ridge()
    ridge.fit(X, y)
    result = [(x, y) for x, y in zip(features_name[1:], ridge.coef_)]
    result2 = sorted(result, key=lambda x: abs(x[1]), reverse=True)


    logistic = LogisticRegression()
    logistic.fit(X, y)

    result = [(x, y) for x, y in zip(features_name[1:], logistic.coef_[0])]
    result3 = sorted(result, key=lambda x: abs(x[1]), reverse=True)




    return ([x[0] for x in result1 if abs(x[1])>0.0000000000001],
            [x[0] for x in result2 if abs(x[1])>0.0000000000001],
            [x[0] for x in result3 if abs(x[1])>0.0000000000001])

def run(csvfile,logger):
    logger.info('linear model start...')
    feature_list = lasso(csvfile)

    logger.info('linear model end.')
    return feature_list
#
# filepath =  r'J:\多设备共享\work\MRMD2.0-github\mixfeature_frequency_DBD.csv'
# result = lasso(filepath)