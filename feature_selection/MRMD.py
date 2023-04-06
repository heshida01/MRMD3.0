import numpy as np
import pandas as pd
import math
from multiprocessing import current_process, cpu_count,Process,Pool
from sklearn.preprocessing import MinMaxScaler
MRMD_models = ['Euc','Cos','Tan']
import sys

def normalize(p,x,mode = "Euc"):
    scale = MinMaxScaler()
    p = np.abs(np.array(p))
    x = np.abs(np.array(x))
    if mode != "Euc" :
        x = 1 - x
    p = scale.fit_transform(p.reshape(-1,1)).reshape(-1)
    x = scale.fit_transform(x.reshape(-1,1)).reshape(-1)

    return list(p),list(x)

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

    numerator =  np.sum(X[:,coli]*X[:,colj])  #分zi
    denominator =  np.sqrt(np.sum(X[:,coli]*X[:,coli])) + np.sqrt(np.sum(X[:,colj]*X[:,colj])) -numerator #分mu
    if denominator ==0:
        return 0

    return numerator / denominator


def calcC(X,coli,colj):

    numerator =  np.sum(X[:,coli]*X[:,colj])  #分zi
    denominator =  np.sqrt(np.sum(X[:,coli]*X[:,coli])) * np.sqrt(np.sum(X[:,colj]*X[:,colj])) #分mu
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
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    mydict = {}
    with Pool(processes=4) as pool:
        task_1 = pool.apply_async(Euclidean, args=(X, n))
        task_2 = pool.apply_async(Tanimoto, args=(X, n))
        task_3 = pool.apply_async(Cosine, args=(X, n))
        task_4 = pool.apply_async(Person, args=(X, y,n))

        mydict.update({"task_1": task_1.get(), "task_2": task_2.get(),"task_3": task_3.get(),"task_4": task_4.get()})

    e=mydict['task_1']
    t=mydict['task_2']
    c=mydict['task_3']
    p=mydict['task_4']


    ###欧氏距离
    mrmrValue=[]
    p, e = normalize(p, e)
    for i,j in zip(p,e):
        mrmrValue.append(i+j)

    mrmr_max=max(mrmrValue)
    mrmrValue = [x / mrmr_max for x in mrmrValue]
    mrmrValue = [(i,j) for i,j in zip(features_name[1:],mrmrValue)]   # features 和 mrmrvalue绑定
    mrmd_e=sorted(mrmrValue,key=lambda x:x[1],reverse=True)  #按mrmrValue 由大到小排序

    ###cos
    mrmrValue=[]
    p, c = normalize(p, c,mode="cos")

    for i,j in zip(p,c):
        mrmrValue.append(i+j)

    mrmr_max=max(mrmrValue)
    mrmrValue = [x / mrmr_max for x in mrmrValue]
    mrmrValue = [(i,j) for i,j in zip(features_name[1:],mrmrValue)]   # features 和 mrmrvalue绑定
    mrmd_c=sorted(mrmrValue,key=lambda x:x[1],reverse=False)  #按mrmrValue 由大到小排序

    ##tan
    mrmrValue=[]
    p, t = normalize(p, t,mode="tan")
    for i,j in zip(p,t):
        mrmrValue.append(i+j)

    mrmr_max=max(mrmrValue)
    mrmrValue = [x / mrmr_max for x in mrmrValue]
    mrmrValue = [(i,j) for i,j in zip(features_name[1:],mrmrValue)]   # features 和 mrmrvalue绑定
    mrmd_t=sorted(mrmrValue,key=lambda x:x[1],reverse=False)  #按mrmrValue 由大到小排序

    mrmd_e = [x[0] for x in mrmd_e]
    mrmd_c = [x[0] for x in mrmd_c]
    mrmd_t =[x[0] for x in mrmd_t]


    logger.info('mrmd end.')
    return mrmd_c,mrmd_e,mrmd_t

def run_s(filecsv,mode):
    #logger.info('mrmd start...')
    X,y,features_name=read_csv(filecsv)
    n=len(features_name)-1

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    #process_list = [Euclidean,Tanimoto,Cosine,Person]

    mydict = {}
    with Pool(processes=2) as pool:
        if mode == "Euclidean":
            task_1 = pool.apply_async(Euclidean, args=(X, n))
        elif mode == "Tanimoto":
            task_1 = pool.apply_async(Tanimoto, args=(X, n))
        elif mode =="Cosine":
            task_1 = pool.apply_async(Cosine, args=(X, n))
        else:
            task_2 = pool.apply_async(Person, args=(X, y,n))

        mydict.update({"task_1": task_1.get(), "task_2": task_2.get()})
    e=mydict['task_1']
    p=mydict['task_2']

    ###欧氏距离
    mrmrValue=[]
    p, e = normalize(p, e)
    for i,j in zip(p,e):
        mrmrValue.append(i+j)

    mrmr_max=max(mrmrValue)
    mrmrValue = [x / mrmr_max for x in mrmrValue]
    mrmrValue = [(i,j) for i,j in zip(features_name[1:],mrmrValue)]   # features 和 mrmrvalue绑定
    mrmd_e=sorted(mrmrValue,key=lambda x:x[1],reverse=True)  #按mrmrValue 由大到小排序


    return  [x[0] for x in mrmd_e]



if __name__ == '__main__':
    # import datetime
    # print(datetime.datetime.now())
    # print ((run('../test.csv',1)))
    # print(datetime.datetime.now())
    print(run_s(sys.argv[1],sys.argv[2]))

