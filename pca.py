# !/usr/bin/env python3
# -*- coding=utf-8 -*-

### 消除警告
def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import argparse
import sklearn.metrics
import time
import logging
import os,sys
from scipy.io import arff
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from format import pandas2arff
from sklearn.manifold import TSNE
from math import ceil
from sklearn.preprocessing import LabelBinarizer,MinMaxScaler
from feature_rank import feature_rank
from sklearn.cluster import  *
from utils.eigen_decomposition import ed_run
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

np.random.seed(1)
def arff2csv_notfile(file):
    with open(file,"r+") as f:
        conent = []
        for line in  f.readlines():
            if line:
                if "{" in line:
                    line = line.replace("{"," {")
                    conent.append(line)
                else:
                    conent.append(line)
    with open(file,"w+") as f:
        for line in  conent:
            f.write(line)

    data = arff.loadarff(file)
    df = pd.DataFrame(data[0])
    df['class'] = df['class'].map(lambda x: x.decode())

    # eg: 0  1    2     3     4  mean =>>  mean   0     1     2    3    4 in dataframe
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    print(df.columns)

arff2csv_notfile("test.arff")