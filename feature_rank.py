import sys
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import load_svmlight_file
from feature_selection.ANOVA import anova
from feature_selection.chisquare import chi2_
from feature_selection.recursive_feature_elimination import ref_
# from  feature_selection.F_value import f_value
from  feature_selection.linear_model import linear_model
from feature_selection.tree_feature_importance import tree_models
from feature_selection.mic import ic
from feature_selection.variance_threshold import vt
import pandas as pd
import numpy as np
from feature_selection import ANOVA
import feature_selection.MRMD
import feature_selection.mrmr
import feature_selection.mic
import feature_selection.ANOVA
import  feature_selection.linear_model
import  feature_selection.chisquare
import feature_selection.recursive_feature_elimination
import feature_selection.tree_feature_importance
import networkx as nx
from  feature_rank_utils.trustrank import trustrank
from feature_rank_utils.leaderank import  leaderrank
from config import methods,lightversion_combination
import sys

def node2edge(nodeOrder_des):  # 特征 大-》小
    edges = []
    for onetype_featureSelection in nodeOrder_des:
        edges += [(onetype_featureSelection[i + 1], onetype_featureSelection[i]) for i, x in
                  enumerate(onetype_featureSelection) if i < len(onetype_featureSelection) - 1]
    ###去重
    # edges = [elem for elem in edges if elem not in edges]
    edges = sorted(set(edges), key=lambda x: edges.index(x))
    return edges


def feature_rank(file,logger,rank_method = None,config =False,light=False):
    if light.lower() == "false":
        if len(methods) == 1 and config.lower() =="true":
            method = str(methods[0])
            if method == "1":
                feature_list = feature_selection.ANOVA.run(file,logger)
            elif method == "2":
                feature_list = feature_selection.chisquare.run(file, logger)
            elif method == "3" or method == "4" or method == "5":
                MI,NMI,MIC = feature_selection.mic.run(file,logger)
                if method == "3":
                    feature_list = MI
                elif method == "4":
                    feature_list = NMI
                elif method == "5":
                    feature_list = MIC
            elif method == "6" or method == "7" or method == "8":
                mrmd_c, mrmd_e, mrmd_t = feature_selection.MRMD.run(file,logger)
                if method == "6":
                    feature_list = mrmd_c
                elif method == "7":
                    feature_list = mrmd_t
                elif method == "8":
                    feature_list = mrmd_e
            elif method == "9" or method == "10" :
                fcq,fcd= feature_selection.mrmr.run(file,logger)
                if method == "9":
                    feature_list = fcd
                elif method == "10":
                    feature_list = fcq
            elif method == "11" or method == "12" or method == "13":
                ref1, ref2, ref3 = feature_selection.recursive_feature_elimination.run(file, logger)
                if method == "11":
                    feature_list = ref1
                elif method == "12":
                    feature_list = ref2
                elif method == "13":
                    feature_list = ref3
            elif method == "14" or method == "15" or method == "16" or method=="17":
                t1,t2,t3,t4 = feature_selection.tree_feature_importance.run(file, logger)
                if method == "14":
                    feature_list = t1
                elif method == "15":
                    feature_list = t2
                elif method == "16":
                    feature_list = t3
                elif method == "17":
                    feature_list = t4
                elif method == "18":
                    feature_list = t3
                elif method == "19":
                    feature_list = t4
            elif method == "20" or method == "21" or method == "22":
                lasso,ridge,logistic = feature_selection.linear_model.run(file, logger)
                if method == "20":
                    feature_list = lasso
                elif method == "21":
                    feature_list = ridge
                elif method == "22":
                    feature_list = logistic
            else:
                print("error,this method is not found ! !")
                sys.exit(-1)
        elif config.lower() == "true":
            methodskv = {"1":"ANOVA","2":"CHis Square","3":"MI","4":"NMI","5":"MIC",
                         "6":"MRMD-Eu","7":"MRMD-Cos","8":"MRMD-Tan","9":"FCQ","10":"FCD",
                         "11":"rfe LogisticRegression","12":"rfe SVM","13":"ref DecisionTree",
                         "14": "RandomForest","15":"DecisionTree","16":"GradientBoosting","17":"ExtraTrees","18":"Adaboost","19":"LGBM",
                         "20": "Lasso","21":"Ridge","22":"Elasticnet"}
            print(f"feature rank ({[methodskv[str(x)] for x in methods]}) result")
            print("################")

        ANOVA_data = feature_selection.ANOVA.run(file,logger)
        chi2_data = feature_selection.chisquare.run(file, logger)
        MI,NMI,MIC = feature_selection.mic.run(file,logger)
        mrmd_c, mrmd_e, mrmd_t = feature_selection.MRMD.run(file,logger)
        fcq,fcd= feature_selection.mrmr.run(file,logger)
        lasso,ridge,logistic = feature_selection.linear_model.run(file, logger)
        ref1,ref2,ref3= feature_selection.recursive_feature_elimination.run(file,logger)
        t1, t2, t3, t4,t5,t6 = feature_selection.tree_feature_importance.run(file, logger)

        feature_rank_result = [ANOVA_data,chi2_data,MI,NMI,MIC,mrmd_c,mrmd_e,mrmd_t,fcq,fcd,ref1,ref2,ref3,t1,t2,t3,t4,t5,t6,lasso,ridge,logistic,]


        all_methods ={str(i+1):x for i,x in enumerate(feature_rank_result)}

        if str(config).lower() == "true":
            feature_rank_result = []
            for  key in methods:
                feature_rank_result.append(all_methods[str(key)])
    else:
        ANOVA_data = feature_selection.ANOVA.run(file, logger)
        chi2_data = feature_selection.chisquare.run(file, logger)
        MI, NMI, MIC = feature_selection.mic.run(file, logger)
        mrmd_c, mrmd_e, mrmd_t = feature_selection.MRMD.run(file, logger)
        lasso, ridge, logistic = feature_selection.linear_model.run(file, logger)

        feature_rank_result = [ANOVA_data, chi2_data, MI, NMI, MIC, mrmd_c, mrmd_e, mrmd_t,
                                lasso, ridge, logistic, ]
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
