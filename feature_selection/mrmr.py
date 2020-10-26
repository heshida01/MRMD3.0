# import os
# import pandas as pd
# import  subprocess
# import sys
#
#
# def process_terminal_info(info:str):
#
#     max_rel = info.split('***')[2].split()[6::4]
#     mrmr= info.split('***')[4].split()[6::4]
#     return mrmr,max_rel
# #http://home.penglab.com/proj/mRMR/
# def mRMR(filecsv):
#     df = pd.read_csv(filecsv,engine='python').dropna(axis=1)
#     n = len(df.columns) - 1
#
#     if 1000>=n>500:
#         n = 500
#     elif 4500>=n>1000:
#         n = 300
#     elif n>=4500:
#         n = 100
#
#     dirname_ = os.path.dirname(__file__)
#     if sys.platform =="linux":
#         if not os.access("chmod 777 {}/util/mrmr".format(dirname_), os.X_OK):  # 如果没有执行权限
#             cmd0 = "chmod +x {}/util/mrmr".format(dirname_)
#             subprocess.Popen(cmd0.split())
#         cmd = "{}/util/mrmr -i {} -n {} ".format(dirname_,filecsv,n)
#     elif sys.platform == "win32":
#         cmd = "{}/util/mrmr_win32.exe -i {} -n {} ".format(dirname_,filecsv,n)
#     elif sys.platform == 'darwin':
#         cmd = "{}/util/mrmr_osx_maci_leopard -i {} -n {} ".format(dirname_,filecsv,n) # mac mrmr_osx_maci_leopard
#
#     terminal_info = subprocess.Popen(cmd.split(), universal_newlines=True,stdout=subprocess.PIPE)
#     mrmr,max_rel = process_terminal_info(terminal_info.communicate()[0])
#     return  mrmr,max_rel
#
# def run(filecsv,logger):
#     try:
#         logger.info('mRMR start...')
#         result = mRMR(filecsv)
#         logger.info('mRMR end.')
#     except:
#         return []
#     else:
#         return result
#
# if __name__ == '__main__':
#     filecsv = '../mixfeature_frequency_DBD.csv'
#     # df = pd.read_csv(filecsv).dropna(axis=1)
#     # n = len(df.columns)-1
#     #
#     # a  =py_mrmr('../mixfeature_frequency_DBD.csv', len(df.columns)-1)
#     # print(a)
#     info = mRMR(filecsv)
#     print(info)
#
#
#
import  numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from operator import itemgetter
import pandas as pd

def entropy(x):

    _, count = np.unique(x, return_counts=True, axis=0)
    prob = count/len(x)
    return np.sum((-1) * prob * np.log2(prob))


def joint_rntropy(y, x):

    yx = np.c_[y, x]
    return entropy(yx)


def conditional_entropy(y, x):

    return joint_rntropy(y, x) - entropy(x)


def mutual_information(x, y):

    return (entropy(x) - conditional_entropy(x, y))

class MRMR():


    def __init__(self, n_features=20, k_max=None):
        self.n_features = n_features
        self.k_max = k_max

    @staticmethod
    def _mutual_information_target(X, y):


        mi_vec = []
        for x in X.T:
            mi_vec.append(mutual_information(x, y))

        return sorted(enumerate(mi_vec), key=itemgetter(1), reverse=True)

    def _handle_fit(self, X, y, threshold=0.8):
        """ handler method for fit """

        ndim = X.shape[1]
        if self.k_max:
            k_max = min(ndim, self.k_max)
        else:
            k_max = ndim

        ## TODO: set k_max
        k_max = ndim

        # mutual informaton between feature fectors and target vector
        MI_trg_map = self._mutual_information_target(X, y)

        # subset the data down to k_max
        sorted_MI_idxs = [i[0] for i in MI_trg_map]
        X_subset = X[:, sorted_MI_idxs[0:k_max]]

        # mutual information within feature vectors
        MI_features_map = {}

        # Max-Relevance first feature
        idx0, MaxRel = MI_trg_map[0]

        mrmr_map = [(idx0, MaxRel)]
        idx_mask = [idx0]

        MI_features_map[idx0] = []
        for x in X_subset.T:
            MI_features_map[idx0].append(mutual_information(x, X[:, idx0]))

        for _ in range(min(self.n_features - 1, ndim - 1)):

            # objective func
            phi_vec = []
            for idx, Rel in MI_trg_map[1:k_max]:
                if idx not in idx_mask:
                    Red = sum(MI_features_map[j][idx] for j, _ in mrmr_map) / len(mrmr_map)
                    phi = (Rel - Red)
                    phi_vec.append((idx, phi))

            idx, mrmr_val = max(phi_vec, key=itemgetter(1))

            MI_features_map[idx] = []
            for x in X_subset.T:
                MI_features_map[idx].append(mutual_information(x, X[:, idx]))

            mrmr_map.append((idx, mrmr_val))
            idx_mask.append(idx)

        mrmr_map_sorted = sorted(mrmr_map, key=itemgetter(1), reverse=True)
        return [x[0] for x in mrmr_map_sorted]

    def fit(self, X, y,features_name, threshold=0.8):
        x = np.array(X)
        if not 0.0 < threshold < 1.0:
            raise ValueError('threshold value must be between o and 1.')


        discretizer = KBinsDiscretizer(
            n_bins=5, encode='ordinal', strategy='uniform')
        ##read_data
        x = discretizer.fit_transform(x)


        return list(features_name[self._handle_fit(x, y, threshold)])


def mRMR(X,y,features_name):
    if len(X.iloc[1,:])>=30:
        n = 30
    else:
        n = len(X.iloc[1,:])
    mrmr = MRMR(n_features=n)
    return mrmr.fit(X, y, features_name,threshold=0.1)

def run(filecsv,logger):

    logger.info('mRMR start...')

    df = pd.read_csv(filecsv, engine='python')
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    features_name = df.columns[1:]
    result = mRMR(X,y,features_name)
    return result
    logger.info('mRMR end.')