from sklearn.feature_selection import SelectKBest,f_classif
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
anova_model = ['anova']
def anova(file,mode=1):

    dataset = pd.read_csv(file,engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)
    X = dataset[:, 1:]
    y = dataset[:, 0]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model1 = SelectKBest(f_classif, k=1)  # 选择k个最佳特征
    model1.fit_transform(X,y)
    result = [(x,y) for x,y in zip(features_name[1:],model1.scores_)]

    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [x[0] for x in result]

def anova_models(file,mode=1):

    dataset = pd.read_csv(file,engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)
    X = dataset[:, 1:]
    y = dataset[:, 0]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model1 = SelectKBest(f_classif, k=1)  # 选择k个最佳特征
    model1.fit_transform(X,y)
    result = [(x,y) for x,y in zip(features_name[1:],model1.scores_)]

    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [x[0] for x in result]


def run(csvfile,logger):
    logger.info('ANOVA start...')
    feature_list = anova(csvfile)
    logger.info('ANOVA  end.')
    return feature_list

def main(file):
    feature_list = anova(file)
    return feature_list

if __name__ == '__main__':

    file = sys.argv[1]
    feature_list = anova(file)
    print(feature_list)