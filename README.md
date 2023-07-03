# [MRMD3.0](https://github.com/heshida01/MRMD3.0/) - the newest version of MRMD is now available!
# mrmd2.0.py 
[WebServer](http://lab.malab.cn:5001/MRMD2.0/Home) ,  [Chinese version](https://github.com/heshida01/MRMD2.0/blob/master/README_CN.md)
# News  

#### 1. Installation：
We recommend using [miniconda3-4.3.31](https://repo.anaconda.com/miniconda/)(or python3.6), support linux,windows.  


  ```
  pip3 install -r requirements.txt --ignore-installed
  ```  

  ##### note:
  If the installation of a Windows user's mine package fails, download the corresponding version of the WHL file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/) to install it.
  
 #### 2. usage:

 ```
 python3  mrmd2.0.py  -i input.csv -s start_index -e end_index -l Step_length  -o metrics.csv  -c Dimensionalized_dataset.csv
 ```
 
*note: The program can select a certain interval of the data set feature sequence for dimensionality reduction, so you need to select the specified parameter -s, -e. If you want to reduce the dimension of the entire data set, you only need to specify -s 1 , -e -1*

 * -i  the input dataset, supports csv,arff and libsvm 
 
 * -s the location where the user specified interval begins （default 1）
 
 * -e User-specified interval ending position （default -1）
 
 * -l step length （default 1，Larger steps will execute faster, and smaller results will be better.）
 
 * -o  Some indicators of the dimensionality reduction data set 
 
 * -b classifier, default=1, RrandomForest=1, SVM=2, Bayes=3
 
 * -r rank_method, default=1,  PageRank = 1,HITS:Authority = 2,HITS:Hub = 3,LeaderRank = 4,TrustRank=5
 
 * -c  Dimensionalized data set 
 
 The data output by the terminal can be found in the Logs directory. Please find the results in 'Results' folder. 

 #### 3. Example
 * Test.csv is a 150-dimensional dataSet
 * First select a dimension reduction interval (here from the first feature to the 150th feature, that is, the dimension reduction of the entire feature data set, of course, you can also choose one of the other continuous feature intervals)  
 * Step size is set to 1  

For feature selection:
 
```
python3  mrmd3.0.py  -i test.csv -o out.csv
python3  mrmd3.0.py  -i test.arff -o out.arff
python3  mrmd3.0.py  -i test.libsvm -o metrics.csv  -c out.libsvm
```
For whole-transcriptome (gene selection):
```
###prepare dataset,datatype ref :   https://scanpy-tutorials.readthedocs.io/en/latest/plotting/core.html
adata = sc.datasets.pbmc68k_reduced()
###Remove low-quality cells and genes
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata) 
###
data = adata.X
Y = adata.obs['bulk_labels'].values
Y = LabelEncoder().fit_transform(Y)
data = np.hstack((data,Y.reshape(-1,1)))
data = pd.DataFrame(data)
data.to_csv('pbmc68k_reduced.csv',index=False,header=False)
###select top 100 genes
!python3  mrmd3.0.py  -i test.csv -o metrics.csv  -c Dimensionalized_dataset.csv -n 100
```

#### 4. FAQs
* problem1: ERROR: Cannot uninstall 'PyYAML'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.   
solve1: pip install -r requirements.txt  --ignore-installed
*************************
* problem2:  error: command 'x86_64-conda_cos6-linux-gnu-gcc' failed with exit status 1.   
solve2:  conda install gxx_linux-64
#### 5. logs
delete pymrmr  
add rfe , chi2  
add HITS LeaderRank TrustRank

If you have any questions.please contact me (heshida@tju.edu.cn)

## reference:  
He, S., Ye, X., Sakurai, T., & Zou, Q. (2023). MRMD3. 0: A Python Tool and Webserver for Dimensionality Reduction and Data Visualization via an Ensemble Strategy. Journal of Molecular Biology, 168116.

He, S., Guo, F., & Zou, Q. (2020). MRMD2. 0: a python tool for machine learning with feature ranking and reduction. Current Bioinformatics, 15(10), 1213-1221.

Zou, Quan, et al. "A novel features ranking metric with application to scalable visual and bioinformatics data classification." Neurocomputing 173 (2016): 346-354.
