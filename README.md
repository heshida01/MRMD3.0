## [MRMD3.0](http://lab.malab.cn/soft/MRMD3.0/index_en.html)
#### [Spark Version(click here)](https://github.com/heshida01/MRMD3.0/tree/master/spark_version)
#### 1.Installation：
*Download*： [github](https://github.com/heshida01/MRMD3.0)  
MRND3.0 can be used directly in an Anaconda environment without the following command.  
environment(python3):  
  ```
  pip3 install -r requirements.txt --ignore-installed
  ```  
  
 #### 2. parameters:
 |parameters|description|
|:-|:-|  
|-s, --start|start index,  default=1 |   
|-i, --inputfile|input file (require:arff ,csv or libsvm format)|   
|-e, --end|end index, default=-1|  
|-l, --length|step length, default=1|
|-n, --n_dim|mrmd2.0 features top n,default=-1|
|-t, --type_metric|evaluation metric, default=f1 |   
|-m, metrics_file|output the metrics file's name|   
|-o, --outfile|output the dimensionality reduction file's name|   
|-r, --rank_method|the rank method for features,choices=["PageRank","Hits_a","Hits_h","LeaderRank","TrustRank"],default="PageRank"|   
|——————————————————|————————————————| 
 #### 3.Example

 ```
python3  mrmd3.0.py  -i test.csv -o out.csv -r PageRank
python3  mrmd3.0.py  -i test.csv -o out.csv -r LeaderRank
python3  mrmd3.0.py  -i test.csv -o out.csv -r TrustRank
python3  mrmd3.0.py  -i test.csv -o out.csv -r Hits_a
python3  mrmd3.0.py  -i test.csv -o out.csv -r Hits_h
 ```
 #### 4. Detail of the feature selection 
|method|the number of the implement method|
|:-|:-|  
|anova|*1 f_classif |   
|chisquare|*1  chi2|   
|F value|*1  f_regression|  
|linear model|*3 Lasso,LogisticRegression,Ridge|
|mutual inforation |*3 MI NMI MIC|
|mrmd|*3 pearson+Euclidean/Tanimoto/Cosine |   
|mrmr|*1 miq|   
|recursive feature elimination|*1 ComplementNB|   
|tree_feature_importance|*3 DecisionTreeClassifier,RandomForestClassifier,GradientBoostingClassifier|   

contact heshida@tju.edu.cn


