import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import *

from itertools import product
from scipy.stats import pearsonr
from multiprocessing import pool


# distance_paras = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
#         'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
#         'jaccard', 'jensenshannon', 'kulsinski', 'matching',
#         'minkowski', 'rogerstanimoto', 'russellrao',
#         'sokalmichener', 'sokalsneath', 'sqeuclidean']


distance = [braycurtis,canberra,chebyshev,cityblock,correlation,cosine,euclidean,jensenshannon,mahalanobis,
                  minkowski,seuclidean,sqeuclidean]

dissimilarity = [dice,hamming,jaccard,kulsinski,rogerstanimoto,russellrao,sokalmichener,sokalsneath,yule,pearsonr]

def pearsoncc(x,y):
    return abs(pearsonr(x,y)[0])

def distance_xy(x,y,distance_metric):

    try:
        # if distance_metric=="mahalanobis":
        #     X = X.T
      #  d = pdist(X, distance_metric)
        d = distance_metric(x,y)

    except:
        return 0

    return d

def mrmd_score(file):
    df = pd.read_csv(file)

    mrmd_distance_list = []
    mrmd_dism_list = []
    mrmd_result_list = []
    for metric in distance:

        sort_result_distance = []
        sort_result_pearson = []

        x = df.iloc[:,1:].values
        y = df.iloc[:,0].values
        n = len(x[0,:])
        distance_matrix = np.zeros([n,n])
        features_name = df.columns[1:]

        for i in range(n):
            for j in range(n):
                if i <= j:
                    #print(distance_xy(x[:,i],x[:,j],distance_metric="se"))
                    distance_matrix[i,j] = distance_xy(x[:,i],x[:,j],distance_metric=metric)
                    distance_matrix[j,i] = distance_matrix[i,j]


        # distance

        for i in range(n):
            sort_result_distance.append(np.sum(distance_matrix[:,i])/n)

        mrmd_distance_list.append(sort_result_distance)
    # dissimilarity
    for metric in dissimilarity:
        sort_result_pearson = []
        for i in range(n):
            #print(metric)
            #print(distance_xy(x[:,i],y,distance_metric=metric))
            if metric.__name__ == "pearsonr" :
                sort_result_pearson.append(abs(pearsonr(x[:, i], y)[0]))
                continue
            sort_result_pearson.append(distance_xy(x[:,i],y,distance_metric=metric))
        #print("sort",len(sort_result_pearson))
        mrmd_dism_list.append(sort_result_pearson)
    ###归一化
    for sort_result_distance,sort_result_pearson in product(mrmd_distance_list,mrmd_dism_list):
        if max(sort_result_distance) != min(sort_result_distance):
            sort_result_distance = (np.array(sort_result_distance)-min(sort_result_distance))/(max(sort_result_distance)-min(sort_result_distance))
        else:

            sort_result_distance = np.zeros(len(sort_result_distance))
        if max(sort_result_pearson) != min(sort_result_pearson):
            sort_result_pearson = (np.array(sort_result_pearson)-min(sort_result_distance))/(max(sort_result_pearson)-min(sort_result_pearson))
        else:

            sort_result_pearson = np.zeros(len(sort_result_pearson))

        mrmd = sort_result_pearson + sort_result_distance
        mrmd_result = sorted([(i, j) for i, j in zip(features_name, mrmd)],key=lambda x:x[1],reverse=True)

        mrmd_result_list.append([x for x,y in mrmd_result])

    return mrmd_result_list

def mrmd_score3(file):
    df = pd.read_csv(file)
    mrmd_result_list = []
    for metric in distance_paras:
      #  print(metric)
        sort_result_distance = []
        sort_result_pearson = []

        x = df.iloc[:,1:].values
        y = df.iloc[:,0].values
        n = len(x[0,:])
        distance_matrix = np.zeros([n,n])
        features_name = df.columns[1:]


        distance_matrix = cdist(x.T,x.T,metric)
        # distance
        for i in range(n):
            sort_result_distance.append(np.sum(distance_matrix[i,:])/n)
        # pear
        for i in range(n):
            sort_result_pearson.append(pearsoncc(x[:,i],y))

        if max(sort_result_distance) != min(sort_result_distance):
            sort_result_distance = (np.array(sort_result_distance)-min(sort_result_distance))/(max(sort_result_distance)-min(sort_result_distance))
        else:
            sort_result_distance = np.zeros(len(sort_result_distance))
        if max(sort_result_pearson) != min(sort_result_pearson):
            sort_result_pearson = (np.array(sort_result_pearson)-min(sort_result_distance))/(max(sort_result_pearson)-min(sort_result_pearson))
        else:
            sort_result_pearson = np.zeros(len(sort_result_pearson))

        ###bing
        mrmd = sort_result_pearson + sort_result_distance
        mrmd_result = sorted([(i, j) for i, j in zip(features_name, mrmd)],key=lambda x:x[1],reverse=True)
     #   print(mrmd_result)
        mrmd_result_list.append([x for x,y in mrmd_result])

    return mrmd_result_list

if __name__ == '__main__':
    import time
    t1 = time.time()
    mrmd_result = mrmd_score("dna.csv")
    print(mrmd_result)
    print(time.time() - t1)

# 6829