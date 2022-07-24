import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

def elbow(X_T,y,file=None,conf=0.95):
    # dataset= pd.read_csv('test.csv')
    # features_name = dataset.columns.values.tolist()
    # dataset = np.array(dataset)
    X = X_T
    #y = dataset[:, 0]
    # '利用SSE选择k'
    SSE = []  # 存放每次结果的误差平方和

    best_K= 0
    for k in range(1, len(X[0,:])):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(X)
        SSE.append(estimator.inertia_)
        best_K = k
        if estimator.inertia_ / max(SSE) <= (1-conf):
            break
    print(best_K)
    # XX = range(1,  len(X[0,:]))
    # plt.xlabel('k')
    # plt.ylabel('SSE')
    # plt.plot(XX, SSE, 'o-')
    # plt.show()

def Silhouette_Coefficient(X_T,y):
    X =X_T
    best_K= 0
    slhouette_cefficients = []

    for k in range(1, len(X[0,:])):
        kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X_T)
        labels = kmeans_model.labels_
        score = metrics.silhouette_score(X_T, labels, metric='sqeuclidean')
        slhouette_cefficients.append(score)

    return slhouette_cefficients.index(max(slhouette_cefficients))



def pca():
    pass

pca


