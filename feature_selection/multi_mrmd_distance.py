import numpy as np
import pandas as pd
from scipy.spatial.distance import *
from itertools import product
from scipy.stats import pearsonr
import multiprocessing
from scipy.stats import pearsonr
distance = [braycurtis, canberra, chebyshev, cityblock, correlation, cosine, euclidean, jensenshannon, mahalanobis,
            minkowski, seuclidean, sqeuclidean]

dissimilarity = [dice, hamming, jaccard, kulsinski, rogerstanimoto, russellrao, sokalmichener, sokalsneath, yule, pearsonr]


def pearsoncc(x, y):
    return abs(pearsonr(x, y)[0])


def distance_xy(x, y, distance_metric):
    try:
        d = distance_metric(x, y)
    except:
        return 0
    return d


def distance_xy_multi(a):
    x, y, distance_metric  = a[0],a[1],a[2]
    try:
        # if distance_metric=="mahalanobis":
        #     X = X.T
        #  d = pdist(X, distance_metric)
        d = distance_metric(x, y)
    except:
        return 0
    return d

def multi_mrmd(x, y, metric, features_name):
    n = 20
    sort_result_pearson = []
    sort_result_distance = []
    distance_matrix = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i <= j:
                # print(distance_xy(x[:,i],x[:,j],distance_metric="se"))
                distance_matrix[i, j] = distance_xy(x[:, i], x[:, j], distance_metric=metric)
                distance_matrix[j, i] = distance_matrix[i, j]
    # distance

    for i in range(n):
        sort_result_distance.append(np.sum(distance_matrix[i, :]) / n)

    # dissimilarity
    for metric in dissimilarity:
        sort_result_pearson = []
        for i in range(n):
            # print(metric)
            # print(distance_xy(x[:,i],y,distance_metric=metric))
            if metric.__name__ == "pearsonr" :
                sort_result_pearson.append(abs(pearsonr(x[:, i], y)[0]))
                continue
            sort_result_pearson.append(distance_xy(x[:, i], y, distance_metric=metric))
        # print("sort",len(sort_result_pearson))

    ###归一化

    if max(sort_result_distance) != min(sort_result_distance):
        sort_result_distance = (np.array(sort_result_distance) - min(sort_result_distance)) / (
                    max(sort_result_distance) - min(sort_result_distance))
    else:

        sort_result_distance = np.zeros(len(sort_result_distance))
    if max(sort_result_pearson) != min(sort_result_pearson):
        sort_result_pearson = (np.array(sort_result_pearson) - min(sort_result_distance)) / (
                    max(sort_result_pearson) - min(sort_result_pearson))
    else:

        sort_result_pearson = np.zeros(len(sort_result_pearson))

    mrmd = sort_result_pearson + sort_result_distance
    mrmd_result = sorted([(i, j) for i, j in zip(features_name, mrmd)], key=lambda x: x[1], reverse=True)
    return mrmd_result


def mrmd_score2(file):
    df = pd.read_csv(file)

    mrmd_distance_list = []
    mrmd_dism_list = []
    mrmd_result_list = []

    x = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    n = len(x[0, :])

    features_name = df.columns[1:]
    pool = multiprocessing.Pool(4)

    for metric in distance:
        pool

        mrmd_result_list.append([x for x, y in mrmd_result])

    return mrmd_result_list

def distance_xy(x, y, distance_metric):

    try:
        # if distance_metric=="mahalanobis":
        #     X = X.T
        #  d = pdist(X, distance_metric)
        d = distance_metric(x, y)
    except:
        return 0
    return d

def func1(args):

        i,x,  metric , feature_n,distance_matrix= args[0],args[1],args[2],args[3],args[4]

        n = feature_n

        result = []
        for j in range(n):
             if i <= j:
                #print(distance_xy(x[:,i],x[:,j],distance_metric="se"))
                distance_matrix[i,j] = distance_xy(x[:,i],x[:,j],distance_metric=metric)
                distance_matrix[j,i] = distance_matrix[i,j]
        return distance_matrix


def func2(args):
    i, x, metric, feature_n = args[0], args[1], args[2], args[3]

    n = feature_n

    result = []
    for j in range(n):
        if i >= j:
            # print(distance_xy(x[:,i],x[:,j],distance_metric="se"))
            result.append(distance_xy(x[:, i], x[:, j], distance_metric=metric))

    return np.array(result)

def mrmd_score(file,n_jobs):
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

        pool = multiprocessing.Pool(n_jobs)
        paras = [(i,x,metric,n) for i in range(n)]

        result = pool.map(func2, paras)

        # distance

        for i in range(n):
            for j in range(n):
                if i>=j:
                    distance_matrix[i,j] = result[i][j]
                else:
                    distance_matrix[i,j] = distance_matrix[j,i]
        for i in range(n):
            sort_result_distance.append(np.sum(distance_matrix[i,:])/n)

        mrmd_distance_list.append(sort_result_distance)
    # dissimilarity
    for metric in dissimilarity:
        sort_result_pearson = []
        for i in range(n):
            if metric.__name__ == "pearsonr" :
                sort_result_pearson.append(abs(pearsonr(x[:, i], y)[0]))
                continue
            sort_result_pearson.append(distance_xy(x[:,i],y,distance_metric=metric))

        mrmd_dism_list.append(sort_result_pearson)

    ###归一化
    for sort_result_distance,sort_result_pearson in product(mrmd_distance_list,mrmd_dism_list):
        if max(sort_result_distance) != min(sort_result_distance):
            sort_result_distance = (np.array(sort_result_distance)-min(sort_result_distance))/(max(sort_result_distance)-min(sort_result_distance))
        else:
            continue
            sort_result_distance = np.zeros(len(sort_result_distance))
        if max(sort_result_pearson) != min(sort_result_pearson):
            sort_result_pearson = (np.array(sort_result_pearson)-min(sort_result_distance))/(max(sort_result_pearson)-min(sort_result_pearson))
        else:
            continue
            sort_result_pearson = np.zeros(len(sort_result_pearson))

        mrmd = sort_result_pearson + sort_result_distance
        mrmd_result = sorted([(i, j) for i, j in zip(features_name, mrmd)],key=lambda x:x[1],reverse=True)

        mrmd_result_list.append([x for x,y in mrmd_result])

    return mrmd_result_list

if __name__ == '__main__':
    import time

    t1 = time.time()
    mrmd_result = mrmd_score("dna.csv",4)
    print(mrmd_result)
    print(time.time() - t1)