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

np.random.seed(1)
def arff2csv(file):
    data = arff.loadarff(file)
    df = pd.DataFrame(data[0])
    df['class'] = df['class'].map(lambda x: x.decode())

    # eg: 0  1    2     3     4  mean =>>  mean   0     1     2    3    4 in dataframe
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    file_csv = file + '.csv'
    df.to_csv(file_csv, index=None)
    return file_csv


def libsvm2csv(file):
    data = load_svmlight_file(file)
    df = pd.DataFrame(data[0].todense())
    df['class'] = pd.Series(np.array(data[1])).astype(int)

    # eg: 0  1    2     3     4  mean =>>  mean   0     1     2    3    4 in dataframe
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    file_csv = file + '.csv'
    df.to_csv(file_csv, index=None)

    return file_csv


if __name__ == '__main__':
    inputfile = sys.argv[1]
    method = sys.argv[2].lower()

    mode = sys.argv[3]
    print(sys.argv)
    if method == "anova":
        if mode == "1" :
            feature_list = anova(inputfile,mode)
    elif method == "chi2":
        if mode == "1":
            feature_list = chi2_(inputfile,mode)
    # elif method == "f_value":
    #     if mode == "1":
    #         feature_list = f_value(inputfile,mode)
    elif method == "linear_model":
            feature_list = linear_model(inputfile,mode)
    elif method == "tree_model":
            feature_list = tree_models(inputfile,mode)
    elif method == "ref":
            feature_list = ref_(inputfile,mode)
    elif method == "ic":
            feature_list = ic(inputfile,mode)
    elif method == "variance_threshold":
            feature_list = vt(inputfile,mode)


