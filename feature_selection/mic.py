# from minepy import MINE
# import numpy as np
# import pandas as pd
# from multiprocessing import Pool,Manager
# import  psutil
# import datetime
#
# def readData(file):
#     dataset=pd.read_csv(file,engine='python').dropna(axis=1)
#     feature_name = dataset.columns.values.tolist()
#     dataset=np.array(dataset)
#     #print(feature_name)
#     return (dataset,feature_name)
#
# def multi_processing_mic(datas):
#     n=psutil.cpu_count(logical=False)
#     n = 1
#     pool=Pool()
#     manager = Manager()
#     dataset = datas[0]
#     features_name = datas[1][1:]  # 去掉label的名字
#
#     mic_score=manager.dict()
#     features_and_index = manager.dict()
#     features_queue = manager.Queue()
#     i = 1
#
#     for name in features_name:
#         features_and_index[name]=i
#         features_queue.put(name)
#         i+=1
#     for i in range (n):
#         pool.apply_async(micscore,(dataset,features_queue,features_and_index,mic_score))
#     pool.close()
#     pool.join()
#     mic_score =[(a,b) for a,b in mic_score.items()]
#     mic_score = sorted(mic_score,key = lambda x:x[1],reverse=True)
#     mic_features =[x[0] for x in mic_score]
#     return mic_features,features_name
# def micscore(dataset,features_queue,features_and_index,mic_score):
#     #print('. ',end='')
#
#     mine = MINE(alpha=0.6, c=15)
#     Y = dataset[:,0]
#
#     while not features_queue.empty():
#         name = features_queue.get()
#         i = features_and_index[name]
#         X=dataset[:,i]
#
#         mine.compute_score(X, Y)
#         score=mine.mic()
#         mic_score[name]= score
#
#     return mic_score

# def run(filecsv,logger):
#     logger.info('mic start...')
#     datas = readData(filecsv)
#     'mic,features_name = micscore(datas)'
#     mic, features_name=multi_processing_mic(datas)
#     #print()
#
#     logger.info('mic end.')
#     return mic,list(features_name)

from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from minepy import MINE
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import metrics
def MI(X,y,features_name):
    np.random.seed(1)
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
    result1 = MI(X,y,features_name=X_y_features_name)
    result2 = NMI(X, y, features_name=X_y_features_name)
    result3 = MIC(X,y,features_name=X_y_features_name)
    logger.info('MI end.')

    return (result1,result2,result3)


if __name__ == '__main__':
    pass



