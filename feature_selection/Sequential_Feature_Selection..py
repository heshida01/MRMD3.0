from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
X, y = load_wine(return_X_y=True, as_frame=True)
n_features = 3
model = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(model,
                                n_features_to_select = n_features,
                                direction='forward') #Try 'backward'
sfs.fit(X, y)
print("Top {} features selected by forward sequential selection:{}"\
      .format(n_features, list(X.columns[sfs.get_support()])))


