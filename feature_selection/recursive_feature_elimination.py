from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.naive_bayes import ComplementNB

def ref_(file):
    dataset = pd.read_csv(file,engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()
    dataset = np.array(dataset)
    X = dataset[:, 1:]
    y = dataset[:, 0]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    estimator = LinearSVC(random_state=1)
    selector = RFE(estimator=estimator, n_features_to_select=1)
    selector.fit_transform(X, y)

    #print(list(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:])))
    #result = sorted(result, key=lambda x: x[1], reverse=True)
    result1 = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))


    estimator = LogisticRegression(random_state=1)
    selector = RFE(estimator=estimator, n_features_to_select=1)
    selector.fit_transform(X, y)
    result2 = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))

    estimator = RandomForestClassifier(random_state=1)
    selector = RFE(estimator=estimator, n_features_to_select=1)
    selector.fit_transform(X, y)
    result3 = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))


    estimator = GradientBoostingClassifier(random_state=1)
    selector = RFE(estimator=estimator, n_features_to_select=1)
    selector.fit_transform(X, y)
    result4= sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))

    estimator = ComplementNB()
    selector = RFE(estimator=estimator, n_features_to_select=1)
    selector.fit_transform(X, y)
    result5 = sorted(zip(map(lambda x: round(x, 4), selector.ranking_), features_name[1:]))

    return ([x[1] for x in result1],
            [x[1] for x in result2],
            [x[1] for x in result3],
            [x[1] for x in result4],
            [x[1] for x in result5],)


def run(csvfile,logger):
    logger.info('ref start...')
    feature_list = ref_(csvfile)
    logger.info('ref end.')
    return feature_list

if __name__ == '__main__':
    res = ref_('../mixfeature_frequency_DBD.csv')
    print(res)