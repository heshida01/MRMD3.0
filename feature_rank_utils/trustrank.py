import tempfile
import os
from feature_rank_utils.PageRank.main import run
import time
import platform
def TrustRank_data_map(feature_order_list_elem):

    return (features_number_g[feature_order_list_elem[0]],features_number_g[feature_order_list_elem[1]])

def trustrank(features_number:dict,feature_order_list:list):
    for key in features_number:
        features_number[key] = features_number[key]-1
    global  features_number_g
    features_number_g = features_number
    feature_len = len(features_number_g)
    osName = platform.system()
    if(osName == 'Windows'):
        tmp = tempfile.NamedTemporaryFile(delete=False)
    else:
        tmp = tempfile.NamedTemporaryFile(delete=True)
    with open(tmp.name,'w+') as fp:

        TrustRank_data = list(map(TrustRank_data_map, feature_order_list))

        for x in TrustRank_data:
           
            content = str(x[0]) + ' ' + str(x[1]) + '\n'
            
            fp.write(content)
        fp.seek(0)
        #print('###')
        #print(str(fp.readlines()))

        location_of_the_edge_file = tmp.name
        number_of_nodes_in_web_graph = feature_len

        #print("tmp.name",tmp.name)
        tr = run(location_of_the_edge_file, number_of_nodes_in_web_graph)
        #print(tr)
        #print(features_number_g)
        tr_featureName2vale = {}
        for key in features_number_g:
            features_number_g[key]=tr[features_number_g[key]]
        #print(features_number_g)
        print(features_number_g)
        return features_number_g
if __name__ == '__main__':
    features_number = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12
        , 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
    feature_order_list = [('A', 'S'), ('E', 'A'), ('N', 'E'), ('G', 'N'), ('L', 'G'), ('V', 'L'), ('P', 'V'),
                          ('W', 'P'), ('C', 'W'), ('Y', 'C'), ('R', 'Y'), ('Q', 'R'), ('K', 'Q'), ('T', 'K'),
                          ('H', 'T'), ('I', 'H'), ('M', 'I'), ('D', 'M'), ('F', 'D'), ('V', 'R'), ('S', 'V'),
                          ('T', 'S'), ('A', 'T'), ('P', 'A'), ('E', 'P'), ('C', 'E'), ('K', 'C'), ('Y', 'K'),
                          ('Q', 'Y'), ('N', 'Q'), ('H', 'N'), ('W', 'H'), ('G', 'W'), ('I', 'G'), ('L', 'I'),
                          ('D', 'L'), ('M', 'D'), ('F', 'M'), ('Q', 'T'), ('R', 'Q'), ('N', 'R'), ('L', 'N'),
                          ('W', 'V'), ('P', 'W'), ('A', 'P'), ('M', 'A'), ('G', 'F'), ('D', 'G'), ('E', 'D'),
                          ('H', 'K'), ('Y', 'I'), ('S', 'L'), ('F', 'S'), ('T', 'F'), ('R', 'T'), ('A', 'R'),
                          ('N', 'P'), ('Y', 'N'), ('C', 'Y'), ('E', 'C'), ('I', 'E'), ('G', 'I'), ('K', 'G'),
                          ('V', 'K'), ('Q', 'V'), ('H', 'Q'), ('M', 'H'), ('W', 'D'), ('Q', 'W'), ('E', 'Q'),
                          ('Y', 'E'), ('N', 'C'), ('V', 'N'), ('M', 'P'), ('I', 'M'), ('R', 'I'), ('F', 'R'),
                          ('G', 'T'), ('A', 'K'), ('L', 'A'), ('S', 'D'), ('C', 'R'), ('S', 'C'), ('Y', 'S'),
                          ('V', 'Y'), ('K', 'V'), ('P', 'K'), ('T', 'E'), ('W', 'T'), ('A', 'W'), ('Q', 'A'),
                          ('N', 'H'), ('T', 'R'), ('S', 'T'), ('C', 'S'), ('E', 'Y'), ('Q', 'E'), ('W', 'Q'),
                          ('K', 'A'), ('I', 'K'), ('P', 'I'), ('V', 'P'), ('N', 'V'), ('H', 'L'), ('G', 'M'),
                          ('F', 'G'), ('D', 'F')]

    trustrank(features_number=features_number,feature_order_list=feature_order_list)

