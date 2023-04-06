import  pandas as pd
import  uuid
import os
def combine_pos_neg(pos_out,neg_out):
    df1 = pd.read_table(pos_out, engine='python', header=None)
    df2 = pd.read_table(neg_out, engine='python', header=None)

    #print(df1.head())
    df1.insert(loc=0, column='class', value=1)
    df2.insert(loc=0, column='class', value=0)
    new_pd:pd.DataFrame = pd.concat([df1,df2])
    #print({str(int(x)): f"fea{x}" for x in range(len(new_pd.columns))})
    new_pd.rename(columns={x: f"fea{x}" for x in range(len(new_pd.columns))}, inplace=True)

    new_pd_filename = str(uuid.uuid1())+'.csv'



    savepath = f"{os.path.dirname(os.path.dirname(__file__))}/Temp/{new_pd_filename}"
    new_pd.to_csv(savepath,index=None)
    #print(savepath+'  SUCCESS!')
    return savepath,new_pd

def Process_iFeature_out_PSeInOne(file,inputfile_label_len):
    df = pd.read_table(file,engine='python',header=None)
    #print(df.head())
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.rename(columns={x: f"fea{x}" for x in range(len(df.columns))}, inplace=True)
    labels = []
    for key in inputfile_label_len:
        labels.extend([int(key)]*inputfile_label_len[key])

    # print(labels)
    # print(len(labels))
    # print(len(df))
    df.insert(loc=0,column='class',value=labels)


    new_pd_filename = os.path.basename(file)+ '.csv'
    savepath = f"{os.path.dirname(os.path.dirname(__file__))}/Temp/{new_pd_filename}"

    df.to_csv(savepath,index=None)
    return df,savepath

def Process_iFeature_out(file,inputfile_label_len):
    df = pd.read_table(file,engine='python')
    df.drop(df.columns[[0]], axis=1, inplace=True)
    labels = []
    for key in inputfile_label_len:
        labels.extend([int(key)]*inputfile_label_len[key])

    #print(labels)
    #print(len(labels))
    #print(len(df))
    df.insert(loc=0,column='class',value=labels)


    new_pd_filename =os.path.basename(file) + '.csv'
    savepath = f"{os.path.dirname(os.path.dirname(__file__))}/Temp/{new_pd_filename}"

    df.to_csv(savepath,index=None)
    return df,savepath

def sc_process(combinefile,inputfiles):

    with open(combinefile, 'w') as outfile:
        for fname in inputfiles:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

                outfile.write('\n')
if __name__ == '__main__':

    """    ####test combine_pos_neg
    f2 = r'J:\多设备共享\work\pycharm_remote_server\FMRMD2.0\negative.txt.out'
    f1 = r'J:\多设备共享\work\pycharm_remote_server\FMRMD2.0\positive.txt.out'
    df,df1 = combine_pos_neg(f1,f2)
    #print(df.head())

    """
    poslen = 691
    neglen = 2715 - 692
    df = Process_iFeature_out(r'J:\多设备共享\work\pycharm_remote_server\FMRMD2.0\Temp\positive.txtnegative.txt.out',poslen=691,neglen=2715-692)