import pandas as pd
from feature_selection import ANOVA, mRMD
import feature_selection.mRMD
import feature_selection.mrmr
import feature_selection.mic
import feature_selection.ANOVA
import  feature_selection.linear_model
import  feature_selection.chisquare
import  feature_selection.F_value
import feature_selection.recursive_feature_elimination
import feature_selection.tree_feature_importance
import networkx as nx
from  feature_rank.trustrank import trustrank
from feature_rank.leaderank import  leaderrank



def feature_rank(file,logger,mrmr_length,rank_method):
    ANOVA_data = feature_selection.ANOVA.run(file,logger)
    MI,NMI,MIC = feature_selection.mic.run(file,logger)
    mrmd_c, mrmd_e, mrmd_t = feature_selection.mRMD.run(file,logger)
    mrmr= feature_selection.mrmr.run(file,logger)
    lasso,ridge,logistic = feature_selection.linear_model.run(file, logger)
    chi2_data = feature_selection.chisquare.run(file,logger)
    t1,t2,t3 = feature_selection.tree_feature_importance.run(file,logger)
    f_value = feature_selection.F_value.run(file,logger)
    ref1,ref2,ref3,ref4,ref5= feature_selection.recursive_feature_elimination.run(file,logger)


    def node2edge(*nodeOrder_des):  #特征 大-》小
        edges=[]
        for onetype_featureSelection in nodeOrder_des:
            edges +=[(onetype_featureSelection[i+1],onetype_featureSelection[i]) for  i,x in enumerate(onetype_featureSelection) if i<len(onetype_featureSelection)-1]
        ###去重
        #edges = [elem for elem in edges if elem not in edges]
        edges = sorted(set(edges), key=lambda x: edges.index(x))
        return  edges

    edges=node2edge(mrmd_c,mrmd_e,mrmd_t,ANOVA_data,MI,NMI,MIC,mrmr,lasso,ridge,logistic,chi2_data,t1,t2,t3,f_value,ref1,ref2,ref3,ref4,ref5)
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

    rankresultWithsocre = webpage_rank(features, graph=G,method=rank_method,edges=edges)
    #print("rankresultWithsocre",rankresultWithsocre)
    logger.info("The final  rank is")
    for value in rankresultWithsocre:
        logger.info(str(value[0])+ " : "+str(value[1]))

    #print('features',features_rc)
    return features_rc,rankresultWithsocre

def webpage_rank(features,graph,method,edges):
    if method.lower() == "vote":
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


        # print("pr=",pr_list)
        # print("hr=", h_list)
        # print("ar=", a_list)
        # print("lr=", lr_list)
        # print("tr=", tr_list)

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
        print(rank_result_list)
        return rank_result_list
    elif method.lower() == "pagerank":
        pr = nx.pagerank(graph)
        return sorted(pr.items(),key=lambda x: x[1], reverse=True)
    elif method.lower() == "hits_a":
        h, a = nx.hits(graph)
        return sorted(a.items(), key=lambda x: x[1], reverse=True)
    elif method.lower() == "hits_h":
        h, a = nx.hits(graph)
        return sorted(h.items(), key=lambda x: x[1], reverse=True)
    elif method.lower() == "leaderrank":
        lr = leaderrank(edges)
        #print("leaderrank+++++++++++",lr.items())
        return sorted(lr.items(), key=lambda item: item[1], reverse=True)
    else:   ###trustrank
        tr = trustrank(features,edges)
        return sorted(tr.items(), key=lambda item: item[1], reverse=True)

if __name__ == '__main__':
    pass
