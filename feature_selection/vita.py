from sklearn.datasets import make_classification,make_regression
from sklearn.ensemble import  RandomForestClassifier,RandomForestRegressor
import numpy as np
import pandas as pd

def importance_pvalues(res):
    rf1_feature_importances_, rf2_feature_importances_, rf_fim_mean = res

def vita(X,y):

    res = holdout_rf(X,y)
    janitza_value =  importance_pvalues(res)



def holdout_rf(X,y):
    rf1 = RandomForestClassifier()
    rf2 = RandomForestClassifier()
    rf1.fit(X,y)
    rf2.fit(X,y)

    rf_fim_mean = (rf1.feature_importances_ + rf2.feature_importances_)/2
    return rf1.feature_importances_,rf2.feature_importances_,rf_fim_mean

if __name__ == '__main__':
    # X,y = make_classification(n_samples=100,n_features=2000,n_classes=2)
    #
    # a,b = make_regression()
    # rf1 = RandomForestClassifier(criterion='entropy',oob_score=True)
    # rf1.fit(X,y)
    # print(sum(rf1.feature_importances_==0))
    #
    # rf2 = RandomForestRegressor()
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=500, n_features=1000, n_informative=2,
                           random_state=0, shuffle=False)
    regr = RandomForestRegressor(criterion="mse", n_estimators=500,
                                 n_jobs=-1,ccp_alpha=0.7,oob_score=True)
    regr.fit(X, y)

    print(sum(regr.feature_importances_ < 0))
