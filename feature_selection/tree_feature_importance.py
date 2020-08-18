from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np

def tree_Fimportance(file):
    dataset= pd.read_csv(file)

    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)
    X = dataset[:, 1:]
    y = dataset[:, 0]

    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(X, y)
    lis1 = rf.feature_importances_


    dt = DecisionTreeClassifier()
    dt.fit(X, y)
    lis2 = dt.feature_importances_

    
    gb = GradientBoostingClassifier()
    gb.fit(X, y)
    lis3 = gb.feature_importances_

    result1 = [(x, y) for x, y in zip(features_name[1:], lis1)]
    result1 = sorted(result1, key=lambda x: x[1], reverse=True)
    result2 = [(x, y) for x, y in zip(features_name[1:], lis2)]
    result2 = sorted(result2, key=lambda x: x[1], reverse=True)
    result3 = [(x, y) for x, y in zip(features_name[1:], lis3)]
    result3 = sorted(result3, key=lambda x: x[1], reverse=True)

    return ([x[0] for x in result1],
            [x[0] for x in result2],
            [x[0] for x in result3])


def run(csvfile, logger):
    logger.info('tree feature importance start...')
    feature_list = tree_Fimportance(csvfile)
    logger.info('tree feature importance  end.')
    return feature_list

if __name__ == '__main__':
    pass
