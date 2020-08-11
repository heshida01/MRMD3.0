### 1. 环境

本方法使用PySpark来实现的，worker机器需要安装requirements.txt中的包 
Sprak版本为Spark3.0

### 2. 使用方法：
类似MRMD3.0的单机模式，
```
 ./spark-submit  --master spark://mymaster:7077  --py-files='feature_rank.zip'  spark_mrmd3.0.py  -i protein.csv -r PageRank
```
