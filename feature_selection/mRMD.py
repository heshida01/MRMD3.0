import numpy as np
import pandas as pd
import math
from scipy.spatial import distance

def read_csv(filecsv):
    dataset=pd.read_csv(filecsv,engine='python').dropna(axis=1)
    features_name = dataset.columns.values.tolist()
    dataset=np.array(dataset)

    X=dataset[:,1:]
    y=dataset[:,0]
    return(X,y,features_name)

def calcE(X,coli,colj):
    # sum=0
    # for i in range(len(X)):
    #
    #      sum+=(X[i,coli]-X[i,colj])*(X[i,coli]-X[i,colj])
    sum = np.sum((X[:,coli]-X[:,colj])**2)


    return math.sqrt(sum)

def calcT(X,coli,colj):

    numerator =  np.sum(X[:,coli]*X[:,colj])  #分母
    denominator =  np.sqrt(np.sum(X[:,coli]*X[:,coli])) * np.sqrt(np.sum(X[:,colj]*X[:,colj]))  #分子
    if denominator ==0:
        return 0
    return numerator/denominator

def calcC(X,coli,colj):

    numerator =  np.sum(X[:,coli]*X[:,colj])  #分母
    denominator =  np.sum(X[:,coli]*X[:,coli]) * np.sum(X[:,colj]*X[:,colj])-numerator  #分子
    if denominator ==0:
        return 0
    return numerator/denominator

def Tanimoto(X,n):
    tanimotodata = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i==j:
                tanimotodata[i, j]=0
            else:
                tanimotodata[i,j]=calcT(X,i,j)
                tanimotodata[j,i]=tanimotodata[i,j]
    tan_distance = []

    for i in range(n):
        sum = np.sum(tanimotodata[i, :])
        tan_distance.append(sum / n)

    return tan_distance

def Euclidean(X,n):

    Euclideandata=np.zeros([n,n])

    for i in range(n):
        for j in range(n):
            Euclideandata[i,j]=calcE(X,i,j)
            Euclideandata[j,i]=Euclideandata[i,j]

    Euclidean_distance=[]

    for i in range(n):
        sum = np.sum(Euclideandata[i,:])
        Euclidean_distance.append(sum/n)

    return Euclidean_distance

def Cosine(X,n):

    Cosinedata = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i==j:
                Cosinedata[i, j]=0
            else:
                Cosinedata[i,j]=calcC(X,i,j)
                Cosinedata[j,i]=Cosinedata[i,j]
    Cos_distance = []

    for i in range(n):
        sum = np.sum(Cosinedata[i, :])
        Cos_distance.append(sum / n)

    return Cos_distance

def varience(data,avg1,col1,avg2,col2):

    return np.average((data[:,col1]-avg1)*(data[:,col2]-avg2))

def Person(X,y,n):
    feaNum=n
    #label_num=len(y[0,:])
    label_num=1
    PersonData=np.zeros([n])
    for i in range(feaNum):
        for j in range(feaNum,feaNum+label_num):
            #print('. ', end='')
            average1 = np.average(X[:,i])
            average2 = np.average(y)
            yn=(X.shape)[0]
            y=y.reshape((yn,1))
            dataset = np.concatenate((X,y),axis=1)
            numerator = varience(dataset, average1, i, average2, j);
            denominator = math.sqrt(
                varience(dataset, average1, i, average1, i) * varience(dataset, average2, j, average2, j));
            if (abs(denominator) < (1E-10)):
                PersonData[i]=0
            else:
                PersonData[i]=abs(numerator/denominator)

    return list(PersonData)

def run(filecsv,logger):
    logger.info('mrmd start...')
    X,y,features_name=read_csv(filecsv)
    n=len(features_name)-1


    e=Euclidean(X,n)
    t=Tanimoto(X,n)
    c=Cosine(X,n)
    p = Person(X,y,n)

    ###欧氏距离
    mrmrValue=[]
    for i,j in zip(p,e):
        mrmrValue.append(i+j)

    mrmr_max=max(mrmrValue)
    mrmrValue = [x / mrmr_max for x in mrmrValue]
    mrmrValue = [(i,j) for i,j in zip(features_name[1:],mrmrValue)]   # features 和 mrmrvalue绑定
    mrmd_e=sorted(mrmrValue,key=lambda x:x[1],reverse=True)  #按mrmrValue 由大到小排序

    ###cos
    mrmrValue=[]
    for i,j in zip(p,c):
        mrmrValue.append(i+j)

    mrmr_max=max(mrmrValue)
    mrmrValue = [x / mrmr_max for x in mrmrValue]
    mrmrValue = [(i,j) for i,j in zip(features_name[1:],mrmrValue)]   # features 和 mrmrvalue绑定
    mrmd_c=sorted(mrmrValue,key=lambda x:x[1],reverse=True)  #按mrmrValue 由大到小排序

    ##tan
    mrmrValue=[]
    for i,j in zip(p,t):
        mrmrValue.append(i+j)

    mrmr_max=max(mrmrValue)
    mrmrValue = [x / mrmr_max for x in mrmrValue]
    mrmrValue = [(i,j) for i,j in zip(features_name[1:],mrmrValue)]   # features 和 mrmrvalue绑定
    mrmd_t=sorted(mrmrValue,key=lambda x:x[1],reverse=True)  #按mrmrValue 由大到小排序

    mrmd_e = [x[0] for x in mrmd_e]
    mrmd_c = [x[0] for x in mrmd_c]
    mrmd_t =[x[0] for x in mrmd_t]


    logger.info('mrmd end.')
    return mrmd_c,mrmd_e,mrmd_t
if __name__ == '__main__':
    import datetime
    print(datetime.datetime.now())
    print ((run('/home/heshida/pycharm_tmp2/experimental_data/diabetes.csv',1)))
    print(datetime.datetime.now())


