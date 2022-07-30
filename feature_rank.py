import pandas as pd
import numpy as np
from feature_selection import ANOVA
import feature_selection.MRMD
import feature_selection.mrmr
import feature_selection.mic
import feature_selection.ANOVA
import  feature_selection.linear_model
import  feature_selection.chisquare
# import  feature_selection.F_value
import feature_selection.recursive_feature_elimination
import feature_selection.tree_feature_importance
import networkx as nx
from  feature_rank_utils.trustrank import trustrank
from feature_rank_utils.leaderank import  leaderrank
import random
import math
from collections import OrderedDict
from feature_selection.mrmd_distance import mrmd_score
from feature_selection.multi_mrmd_distance import mrmd_score as mms
random.seed(1)


def distance():
    pass

def node2edge(nodeOrder_des):  # 特征 大-》小
    edges = []
    for onetype_featureSelection in nodeOrder_des:
        edges += [(onetype_featureSelection[i + 1], onetype_featureSelection[i]) for i, x in
                  enumerate(onetype_featureSelection) if i < len(onetype_featureSelection) - 1]
    ###去重
    # edges = [elem for elem in edges if elem not in edges]
    edges = sorted(set(edges), key=lambda x: edges.index(x))
    return edges


def feature_rank(file,logger,rank_method = None,mode = "0",n_jobs = 1):

    def select_feature(feature_list_by_feature_rank: list):


        max_index = len(feature_list_by_feature_rank)

        w1 = 10
        w2 = 7/11
        # x_b = -(w1/w3)+w2  # 29.306930693069308
        y_b = math.pow(w1,w2)  #9.866666666666667


        p_fun = lambda  x: math.pow(-x+w1,w2)

        def p_ex(x,p1 = 1):
            p1 = p1*math.exp(-1*p1*x)

            return x

        select = []
        not_select = []
        p_list = []


        for feature in feature_list_by_feature_rank:
            x = feature_list_by_feature_rank.index(feature)
            x = (x / max_index) * 3
            p = p_ex(x)


            if random.choices([True, False], weights=[p, 1 - p], k=1)[0]:
                select.append(feature)
            else:
                not_select.append(feature)
            p_list.append(p)

        return select, not_select,p_list

    if mode == "1": # fast
        feature_rank_result = mrmd_score(file)
        if int(n_jobs) > 1:
            feature_rank_result = mms(file,n_jobs)
    else:
        ANOVA_data = feature_selection.ANOVA.run(file,logger)
        MI,NMI,MIC = feature_selection.mic.run(file,logger)
        mrmd_c, mrmd_e, mrmd_t = feature_selection.MRMD.run(file,logger)
        miq,mid= feature_selection.mrmr.run(file,logger)
        lasso,ridge,logistic = feature_selection.linear_model.run(file, logger)
        chi2_data = feature_selection.chisquare.run(file,logger)
        t1,t2,t3,t4 = feature_selection.tree_feature_importance.run(file,logger)
        # f_value = feature_selection.F_value.run(file,logger)
        ref1,ref2,ref3= feature_selection.recursive_feature_elimination.run(file,logger)

        feature_rank_result = [ANOVA_data,MI,NMI,MIC,mrmd_c, mrmd_e, mrmd_t,miq,mid,lasso,ridge,logistic,chi2_data,t1,t2,t3,t4,ref1,ref2,ref3]


    if rank_method != None:

        edges = node2edge(feature_rank_result)
        G = nx.DiGraph()
        G.add_edges_from(edges)

        features = {}
        i = 1
        dataset = pd.read_csv(file, engine='python').dropna(axis=1)
        features_name = dataset.columns.values.tolist()[1:]
        for x in features_name:
            features[x] = i
            i += 1
        features_rc = features.copy()

        rankresultWithsocre = webpage_rank(features, graph=G, method=rank_method, edges=edges)
        # print("rankresultWithsocre",rankresultWithsocre)
        logger.info("The final  rank is")
        for value in rankresultWithsocre:
            logger.info(str(value[0]) + " : " + str(value[1]))

        # print('features',features_rc)
        return features_rc, rankresultWithsocre


    select_result = []
    not_select_result = []
    # for select_feature_single_result in [mrmd_c, mrmd_e, mrmd_t, ANOVA_data, MI, NMI, MIC, miq,mid, lasso, ridge, logistic, chi2_data, t1, t2,
    #                   t3, t4,f_value,  ref5]:
    for select_feature_single_result in feature_rank_result:
        if type(select_feature_single_result) != 'NoneType':
            select, not_select ,p_list= select_feature(select_feature_single_result)

            select_result.append(select)
            not_select_result.append(not_select)


    #edges=node2edge(mrmd_c,mrmd_e,mrmd_t,ANOVA_data,MI,NMI,MIC,mrmr,lasso,ridge,logistic,chi2_data,t1,t2,t3,f_value,ref1,ref2,ref3,ref4,ref5)
    edges = node2edge(select_result)
    G = nx.DiGraph()
    G.add_edges_from(edges)
    features = OrderedDict()
    i = 1
    dataset = pd.read_csv(file, engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()[1:]
    for x in features_name:
        features[x] = i
        i += 1
    features_rc = features.copy()

    #rankresultWithsocre = node_features_rank(features,select_features=select_result,not_select_features=not_select_result)

    rankresultWithsocre = webpage_rank(features, graph=G,method=rank_method,edges=edges)
    #print("rankresultWithsocre",rankresultWithsocre)
    logger.info("The final  rank is")
    for value in rankresultWithsocre:
        logger.info(str(value[0])+ " : "+str(value[1]))

    #print('features',features_rc)
    return features_rc,rankresultWithsocre

def node_features_rank(features,select_features,not_select_features):
    ### generate good and bad features   link
    good_nodes_link = node2edge(select_features)
    bad_nodes_link = node2edge(not_select_features)

    G_good = nx.DiGraph()
    G_good.add_edges_from(good_nodes_link)

    G_bad = nx.DiGraph()
    G_bad.add_edges_from(bad_nodes_link)

    ### 构建M
    nodes_num = len(features)
    M_good = np.zeros((nodes_num, nodes_num))
    M_bad = np.zeros((nodes_num, nodes_num))
 

    for node in G_good.adjacency():
        for key in node[1]:
            # print(key,1/len(node[1]),end=' ')
            M_good[features[key] - 1][features[node[0]] - 1] = 1 / len(node[1])

    for node in G_bad.adjacency():
        for key in node[1]:
            # print(key,1/len(node[1]),end=' ')
            M_bad[features[key] - 1][features[node[0]] - 1] = 1 / len(node[1])

    def R0_fun(select,not_select,features):

        N_select =  sum([len(x) for x in select])
        N_notselect= sum([len(y) for y in not_select])
        N_total = N_select + N_notselect

        select_stat = {}
        not_select_stat = {}

        for x in features.keys():
            select_stat[x] = 0

        for x in features.keys():
            not_select_stat[x] = 0

        select_stat_list  = [y for x in select_features for y in x]

        for x in select_stat_list:
            # if x not in select_stat.keys():
            #     select_stat[x] = 0
            # else:
            #     select_stat[x] += 1
            select_stat[x] += 1
        not_select_stat_list = [y for x in not_select_features for y in x]
        for x in not_select_stat_list:
            # if x not in not_select_stat.keys():
            #     not_select_stat[x] = 0
            # else:
            #     not_select_stat[x] += 1
            not_select_stat[x] += 1

        R0 = np.zeros(len(features))
        for i,x in enumerate(features.keys()):
            # print(i,x,features.keys())
            #
            # if x not in select_stat.keys() :
            #     select_stat[x] = 0
            #     R0[i] = (1/not_select_stat[x])/N_total
            #
            # if x not in not_select_stat.keys():
            #     R0[i] = (select_stat[x])/N_total
            if not_select_stat[x] == 0:
                a = 0
            else:
                a = ((1/not_select_stat[x]))/N_total
            b = (select_stat[x])/N_total
            R0[i] = a + b
        ###softmax
        def softmax(x):

            f_x = np.exp(x) / np.sum(np.exp(x))
            return f_x

        return softmax(R0)
    R0 = R0_fun(select_features,not_select_features,features)

    print(R0)
    a = 5 / 10
    b = 3 / 10

    random_m = ((1 - a - b) / nodes_num) * np.ones((nodes_num))

    R = R0
    for x in range(100):
        R_good = a * M_good.dot(R)
        R_bad = b * M_bad.dot(R)
        R = R_good + R_bad + random_m
        # print(R.reshape(nodes_num))
    #print(R)
    # R = R.reshape(nodes_num)
    features_k_v = features
    for i,x in enumerate(features_k_v.keys()):
         features[x] = R[i]


    return  sorted(features.items(),key=lambda x: x[1], reverse=True)

def webpage_rank(features,graph,method,edges):

    if str(method).lower() == "vote":
        pr = nx.pagerank(graph)
        h, a = nx.hits(graph)
        lr = leaderrank(graph)
        tr = trustrank(features, edges)

        pr_list = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        a_list = sorted(a.items(), key=lambda x: x[1], reverse=True)
        h_list  = sorted(h.items(), key=lambda x: x[1], reverse=True)
        lr_list = sorted(lr.items(), key=lambda item: item[1], reverse=True)
        tr_list = sorted(tr.items(), key=lambda item: item[1], reverse=True)
        order_nos_list = [new_score for new_score in reversed(range(len(pr_list)))]

        pr_list2 = []
        a_list2 = []
        h_list2  = []
        lr_list2 = []
        tr_list2 =  []
        for (p,a,h,l,t,o) in zip(pr_list,a_list,h_list,lr_list,tr_list,order_nos_list):

            p = (p[0],o)
            a = (a[0], o)
            h = (h[0], o)
            l = (l[0], o)
            t = (t[0], o)
            pr_list2.append(p)
            a_list2.append(a)
            h_list2.append(h)
            lr_list2.append(l)
            tr_list2.append(t)


        # print("pr=",pr_list2)
        # print("hr=", h_list2)
        # print("ar=", a_list2)
        # print("lr=", lr_list2)
        # print("tr=", tr_list2)

        rank_result = {}
        for elem in pr_list2:
            if elem[0] not in rank_result.keys():
                rank_result[elem[0]] = 0
                continue
            rank_result[elem[0]] = elem[1]
        for elem2,elem3,elem4,elem5 in zip(h_list2,a_list2,lr_list2,tr_list2):
            rank_result[elem2[0]] += elem2[1]
            rank_result[elem3[0]] += elem3[1]
            rank_result[elem4[0]] += elem4[1]
            rank_result[elem5[0]] += elem5[1]
        rank_result_list = sorted(rank_result.items(), key=lambda item: item[1], reverse=True)
        #print(rank_result_list)
        return rank_result_list
    elif str(method).lower() == "pagerank":
        pr = nx.pagerank(graph)
        return sorted(pr.items(),key=lambda x: x[1], reverse=True)
    elif str(method).lower() == "hits_a":
        h, a = nx.hits(graph)
        return sorted(a.items(), key=lambda x: x[1], reverse=True)
    elif str(method).lower() == "hits_h":
        h, a = nx.hits(graph)
        return sorted(h.items(), key=lambda x: x[1], reverse=True)
    elif str(method).lower() == "leaderrank":
        lr = leaderrank(graph)
        #print("leaderrank+++++++++++",lr.items())
        return sorted(lr.items(), key=lambda item: item[1], reverse=True)
    else:   ###trustrank
        tr = trustrank(features,edges)
        return sorted(tr.items(), key=lambda item: item[1], reverse=True)

if __name__ == '__main__':
    pass
