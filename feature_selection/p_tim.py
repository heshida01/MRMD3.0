# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from boruta import BorutaPy
# from sklearn.datasets import make_classification
# # 导入数据
#
# X,y = make_classification(n_samples=100,n_features=10)
# # 定义随机森林分类器
# rf = RandomForestClassifier(n_jobs=-1,class_weight='balanced',max_depth=5)
# # 设置 Boruta 特征选择的参数
# feat_selector = BorutaPy(rf,n_estimators='auto',verbose=0,random_state=1)
#
# # 发现所有相关的特征-5个特征会被选择
# feat_selector.fit(X, y)
#
# # 查看前五个选择的特征
# feat_selector.support_
#
# # 查看选择的特征的rank
# feat_selector.ranking_
#
# # 用 transform() 过滤掉数据x不相关的特征
# X_filtered = feat_selector.transform(X)
#
# print(feat_selector.support_)
# print(feat_selector.ranking_)