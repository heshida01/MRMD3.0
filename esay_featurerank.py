from feature_selection import  *
import argparse
import collections
from feature_rank_utils import *
feature_methods = collections.OrderedDict()

methods = ['ANOVA',
           'chisquare',
           'f_value',
           'linear model:Lasso','linear model:Ridge','linear model:Logistic',
           'MRMD:Euc','MRMD:Cos','MRMD:Tan',
           'MIC','NMI','MI',
           'MRMR:MID','MRMR:MIQ'
           'tree model:DecisionTree','RandomForest','GradientBoosting','ExtraTreesClassifier'
           'Variance'
           ]

#recursive feature elimination
rfe =    ['LinearSVC',"LinearSVR",
           'ComplementNB','MultinomialNB','BernoulliNB',
            "Lasso","Ridge","ElasticNet",
            "DecisionTreeClassifier","RandomForestClassifier","GradientBoostingClassifier"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", dest='m', type=str,  default=1)
    parser.add_argument("-i", "--inputfile", dest='i', type=str, help="input file", required=True)
    parser.add_argument("-o", "--outfile", dest='o', type=str, help="output the dimensionality reduction file", required=True)
    parser.add_argument("-p", "--parameters", dest='p', type=str, help="the feature rank's parameters file path")

    args = parser.parse_args()

    return args

methods_file = fun_list()[:-1]
methods_i2m = {i:x for i,x in enumerate(methods)}
"""
{0: 'ANOVA',
 1: 'chisquare',
 2: 'f_value',
 3: 'linear_model:Lasso',
 4: 'linear_model:Ridge',
 5: 'linear model:Logistic',
 6: 'MRMD:Euc',
 7: 'MRMD:Cos',
 8: 'MRMD:Tan',
 9: 'MIC',
 10: 'NMI',
 11: 'MI',
 12: 'MRMR:MID',
 13: 'MRMR:MIQ
 14: 'model:DecisionTree',
 14: 'RandomForest',
 15: 'GradientBoosting',
 16: 'ExtraTreesClassifierVariance'}
"""
def run(file,method):

    if method == 0:
        model = ANOVA
    elif method == 1:
        model = chisquare
    elif method == 2:
        model = F_value
    elif method == 3:
        model = linear_model
    elif method == 4:
        model = linear_model
    elif method == 5:
        model = linear_model
    elif method == 6:
        model =  MRMD
    elif method == 7:
        model =  MRMD
    elif method == 8:
        model =  MRMD
    elif method == 9:
        model =  MIC
    elif method == 10:
        model =  NMI
    elif method == 11:
        model =  MI


    print(model.main(file))
if __name__ == '__main__':
    file = 'test.csv'
    run(file,method=0)
