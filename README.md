## [MRMD3.0](http://lab.malab.cn/soft/MRMD3.0/index_en.html)
### Update!  [WebServer(click here)](https://github.com/heshida01/MRMD3.0/tree/master/spark_version)
MRMD3.0 is a method that integrates multiple feature ranking algorithms, which can automatically infer best dimensions and generate data charts.
## Installation：
environment(python3):  
  ```
  pip3 install -r requirements.txt 
  ```  
###### Support OS:  Win,Linux,Mac
 ## Usage:
##### MRMD3.0
```
 python3 mrmd3.0.py [-h] [-s S] -i I [-e E] [-l L] [-n N] [-t T] [-c {RandomForest,SVM,Bayes}] [-o O] [-p P] [-m M] [-j J] [-f F] [-r {PageRank,Hits_a,Hits_h,LeaderRank,TrustRank}]
```

 |parameters|description|
|:-|:-|  
|-s, --start|start index,  default=1 |   
|-i, --inputfile|input file (require:arff ,csv or libsvm format)|   
|-e, --end|end index, default=-1|  
|-l, --length|step length, default=1|   
|-n, --n_dim|mrmd3.0 features top n,default=-1|  
|-t, --type_metric|evaluation metric, default=f1 | 
|-c,--classifier|cross vaildaion classifier r {RandomForest,SVM,Bayes}, default=RandomForest|  
|-m, metrics_file|output the metrics file's name|   
|-o, --outfile|output the dimensionality reduction file's name| 
|-f,--topn| select top n features to chart|  
|-r, --rank_method|the rank method for features,choices=["PageRank","Hits_a","Hits_h","LeaderRank","TrustRank"],default="PageRank"|   
|——————————————————|————————————————| 
 ##### Example

 ```
python3  mrmd3.0.py  -i test.csv -o out.csv 
 ```
 ## feature selection (new)
Users can rank the features of the dataset through the interface of MRMD3.0 using the feature sorting method
Parameters
|feature selection|method|  
|:-|:-|  
|anova|1.anova |  
|VarianceThreshold|1.VarianceThreshold|
|chisquare| 1. chisquare|  
|linear_models|1. lasso, 2. ridge, 3.elasticnet |  
|mutual inforation | 1.MI 2.NMI 3.MIC| 
|mrmr| 1.miq 2.miq|   
|recursive_feature_elimination|1. LogisticRegression, 2.SVM , 3. DecisionTreeClassifier|   
|tree_feature_importance| 1. DecisionTreeClassifier, 2. RandomForestClassifier, 3. GradientBoostingClassifier 4.ExtraTreesClassifier|   
##### example
python mrmd_fs.py  test.csv  linear_models lasso
<br>
contact heshida@tju.edu.cn


