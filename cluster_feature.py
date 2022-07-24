# clustering dataset
# determine k using elbow method
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
def test():
    x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
    x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
    plt.plot()
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.title('Dataset')
    plt.scatter(x1, x2)
    plt.show()
    # create new plot and data
    plt.plot()
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
    colors = ['b', 'g', 'r']
    markers = ['o', 'v', 's']
    # k means determine k
    distortions = []
    K = range(1,len(x1)-1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def clu_gmm(X,features,features_sorted, threshold =0.7 ):
    # [('Q', 0.4135146288735066), ('Y', 0.39893698213692674), ('S', 0.3877355423066138), ('A', 0.37869036611873325), ('G', 0.3585291694093776), ('R', 0.3564785894669215), ('C', 0.33684050001203647), ('N', 0.3128496649686209), ('E', 0.31253485677608217), ('D', 0.3084676327695345), ('I', 0.3069034315358833), ('P', 0.2686084169563175), ('H', 0.2633607086324316), ('K', 0.21717671201197009), ('T', 0.21567079046297372), ('F', 0.20491791357491876), ('L', 0.2039444517457154), ('V', 0.18273791283791718), ('M', 0.15901523300208054), ('W', 0.1585000287076487)]
    # features =Dict([('A', 1), ('C', 2), ('D', 3), ('E', 4), ('F', 5), ('G', 6), ('H', 7), ('I', 8), ('K', 9), ('L', 10), ('M', 11), ('N', 12), ('P', 13), ('Q', 14), ('R', 15), ('S', 16), ('T', 17), ('V', 18), ('W', 19), ('Y', 20)])

    features_name = [x[0]  for x in features.items()]
    features_order_by_name = [x[0] for x in features_sorted]
    features_order_by_key = []
    #
    # features_name = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # features_order_by_name = ['E', 'G', 'C', 'R', 'I', 'A', 'T', 'D', 'Q', 'S', 'K', 'V', 'F', 'H', 'Y', 'P', 'N', 'L', 'M', 'W']
    # features_order_by_key = [3, 5, 1, 14, 7, 0, 16, 2, 13, 15, 8, 17, 4, 6, 19, 12, 11, 9, 10, 18]

    for name in  features_order_by_name:
        features_order_by_key.append(features[name]-1)

    length = len(features_name)

    K = range(1,length)
    distortions = []

    X_T = X.T
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(X_T)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'cosine'), axis=1)) / X.shape[0])
    norm_dist = []
    for x in distortions:
        norm_dist.append(x/max(distortions))
    print(norm_dist)
    if  threshold  == "auto":
        pass
    else:
        index = 0
        for i,v in enumerate(norm_dist):
             if v <= float(threshold):
                 index = i
                 break

    #    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def demo():
    #test()
    from scipy.spatial.distance import cdist
    import matplotlib.pyplot as plt
    import seaborn as sns;
    from sklearn.mixture import GaussianMixture as GMM
    sns.set()
    import numpy as np
    from sklearn.cluster import KMeans

    # https://zhuanlan.zhihu.com/p/81255623
    # 产生实验数据
    from sklearn.datasets.samples_generator import make_blobs

    X, y_true = make_blobs(n_samples=500, centers=88, n_features=100,
                           cluster_std=0.5, random_state=2017)
    K = range(1, X.shape[0] )
    distortions = []
    for k in K:
        gmm = GMM(n_components=k).fit(X)  # 指定聚类中心个数为4
        labels = gmm.predict(X)
        distortions.append(sum(np.min(cdist(X, gmm.means_, 'cosine'), axis=1)) / X.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k gmm')
    plt.show()

    distortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k).fit(X)  # 指定聚类中心个数为4
        #labels = gmm.predict(X)
        distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'sqeuclidean'), axis=1)) / X.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k kmean')
    plt.show()

if __name__ == '__main__':
    demo()