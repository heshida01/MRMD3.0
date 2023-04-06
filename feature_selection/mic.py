from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from minepy import MINE
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import metrics
from multiprocessing import Pool
import sys
#MI_models = ['MI','MIC','NMI']
def MI(X,y,features_name):

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model1 = SelectKBest(mutual_info_classif, k=1)  # 选择k个最佳特征
    model1.fit_transform(X, y)
    result = [(x, y) for x, y in zip(features_name[:], model1.scores_)]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    return [x[0] for x in result]


def MIC(X, y, features_name):
    mic_score = {}

    for name in features_name:
        mine = MINE(alpha=0.6, c=15)
        score = mine.compute_score(X[name], y)
        # score = mine.compute_score(X[name].tolist(), y.tolist())
        score = mine.mic()
        mic_score[name] = score

    mic_score = [(a, b) for a, b in mic_score.items()]
    mic_score = sorted(mic_score, key=lambda x: x[1], reverse=True)
    mic_features = [x[0] for x in mic_score]

    return mic_features


def NMI(X, y, features_name):
    NMI_score = {}
    discretizer = KBinsDiscretizer(
        n_bins=5, encode='ordinal', strategy='uniform')

    for name in features_name:
        x = np.array(X[name]).reshape(-1, 1)
        x = discretizer.fit_transform(x)
        x = x.reshape(-1)
        score = metrics.normalized_mutual_info_score(x, y)
        # score = mine.compute_score(X[name].tolist(), y.tolist())

        NMI_score[name] = score

    NMI_score = [(a, b) for a, b in NMI_score.items()]
    NMI_score = sorted(NMI_score, key=lambda x: x[1], reverse=True)
    NMI_features = [x[0] for x in NMI_score]

    return NMI_features


def run(csvfile, logger):
    logger.info('MI start...')
    df = pd.read_csv(csvfile, engine='python')
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    X_y_features_name = df.columns.tolist()[1:]


    mydict = {}
    with Pool(processes=4) as pool:
        task_1 = pool.apply_async(MI, args=(X,y,X_y_features_name))
        task_2 = pool.apply_async(NMI, args=(X,y,X_y_features_name))
        task_3 = pool.apply_async(MIC, args=(X,y,X_y_features_name))
        mydict.update({"task_1": task_1.get(), "task_2": task_3.get(),"task_3": task_2.get()})

    result1 = mydict['task_1']
    result2 = mydict['task_1']
    result3 = mydict['task_1']
    logger.info('MI end.')

    return (result1,result2,result3)

def ic(file,mode:str):

    df = pd.read_csv(file, engine='python')
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    X_y_features_name = df.columns.tolist()[1:]
    if mode.upper() == "MIC":
        result = MIC(X, y, X_y_features_name)
    elif mode.upper() == "MI":
        result = MI(X, y, X_y_features_name)
    elif mode.upper() == "NMI":
        result = NMI(X, y, X_y_features_name)

    return  result

if __name__ == '__main__':
    file = sys.argv[1]
    df = pd.read_csv(file, engine='python')
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    X_y_features_name = df.columns.tolist()[1:]
    if sys.argv[2] == "MIC":
        result = MIC(X,y,X_y_features_name)
    elif sys.argv[2] == "MI":
        result = MI(X,y,X_y_features_name)
    elif sys.argv[2] == "NMI":
        result = NMI(X,y,X_y_features_name)

    print(result)
