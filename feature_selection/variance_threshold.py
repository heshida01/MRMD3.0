from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
f_value_models = ['f_value']

def vt(file,mode):
    dataset = pd.read_csv(file, engine='python')
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)
    X = dataset[:, 1:]
    y = dataset[:, 0]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model1 = VarianceThreshold() # 选择k个最佳特征
    model1.fit_transform(X, y)
    result = [(x, y) for x, y in zip(features_name[1:], model1.variances_)]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [x[0] for x in result]


def run(csvfile, logger):
    logger.info('F_value start...')
    feature_list = vt(csvfile)
    logger.info('F_value  end.')
    return feature_list

if __name__ == '__main__':
    csvfile, = sys.argv[1]
    print(vt(csvfile,None))

