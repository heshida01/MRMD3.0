### 1. environment 
a. Spark3.0  
b. you need to install the following python packages on the cluster machines：

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
  ```

### 2. example：
The usage simliar to MRMD3.0:
```
 ./spark-submit  --master spark://mymaster:7077  --py-files='feature_rank.zip'  spark_mrmd3.0.py  -i test.csv -r PageRank
```
