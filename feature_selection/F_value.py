from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def f_value(file):
    dataset = pd.read_csv(file, engine='python')
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)
    X = dataset[:, 1:]
    y = dataset[:, 0]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model1 = SelectKBest(f_regression, k=1)  # 选择k个最佳特征
    model1.fit_transform(X, y)
    result = [(x, y) for x, y in zip(features_name[1:], model1.scores_)]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [x[0] for x in result]


def run(csvfile, logger):
    logger.info('F_value start...')
    feature_list = f_value(csvfile)
    logger.info('F_value  end.')
    return feature_list

if __name__ == '__main__':
    pass
