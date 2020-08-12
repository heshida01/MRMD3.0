### 1. environment 
a. Spark3.0  
b. java jdk1.8  
c. python3.6  
d. you need to install the following python packages on the cluster machines：

  ```
  pyspark==3.0.0  
  findspark==1.4.2  
  numpy==1.19.0  
  networkx==2.4  
  pandas==1.0.5  
  minepy==1.2.4  
  matplotlib==3.3.0  
  scipy==1.5.0  
  scikit_learn==0.23.2  
  pyarrow==0.17.1
  ```
##### note:
To facilitate the synchronization of the cluster files, we provide a [Linux script](https://github.com/heshida01/MRMD3.0/blob/master/spark_version/easy_distribution%20_package_demo.sh) here.

### 2. parameters
 |parameters|description|
|:-|:-|  
|-i, --inputfile|input file (require:csv format)|   
|-t, --type_metric|evaluation metric(f1,accuracy,precision,recall,auc), default=f1 |   
|-c, --classifier|classifier(RandomForest,SVM,Bayes) default="RandomForest"|   
|-r, --rank_method|the rank method for features,choices=["PageRank","Hits_a","Hits_h","LeaderRank","TrustRank"],default="PageRank"|   
|——————————————————|————————————————| 

### 3. example：
The usage simliar to MRMD3.0:
```
 ./spark-submit  --master spark://mymaster:7077  --py-files='feature_rank.zip'  spark_mrmd3.0.py  -i test.csv -r PageRank
 ./spark-submit  --master spark://mymaster:7077  --py-files='feature_rank.zip'  spark_mrmd3.0.py  -i test.csv -r Hits_a
```
