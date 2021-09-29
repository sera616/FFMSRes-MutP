# coding: utf-8

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
warnings.filterwarnings("ignore")
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from pandas import DataFrame
from keras.models import load_model
from feature_extract import feature_extract
from feature_generate import feature_generate
import sendemail

def read_text(data_path, jopid):
    seq_path = data_path + jobid + ".txt"
    print(seq_path)
    with open(seq_path, "r") as f:
        lines=f.readlines()
        head = []
        seq_name =[]
        seq = []
        pos=[]
        wtAA = []
        mutAA = []
        for i in range(len(lines)):
            if (lines[i][0]=='>'):
                temp = lines[i].strip()
                head.append(temp)
            else:
                seq.append(lines[i].strip())
        for j in range(len(head)):
            hp = head[j].split('_')[0]
            seq_name.append(hp[1:])
            mp = head[j].split('_')[1]                  
            wtAA.append(mp[0])
            pos.append(mp[1:-1])
            mutAA.append(mp[-1])        
    return (seq_name, seq, pos, wtAA, mutAA)

def main(data_path, model_path, jobid):    
    import pandas as pd
    import numpy
    from numpy import array
    import numpy as np
    from pandas import DataFrame
    from keras.models import load_model
    seq_name, seq, pos, wtAA, mutAA = read_text(data_path,jobid)
    feature_generate(data_path,jobid,seq_name, seq, pos, wtAA, mutAA,)
    feature_extract(data_path,jobid)    
    file_openpath = data_path + "PSPP1D28.csv"
    data = pd.read_csv(file_openpath)
    y_test = data['Label']
    X_test_seq = data.loc[:,'pssm0':'pssm1179']
    X_test_seq_Features = X_test_seq
    X_test_seq_len = len(X_test_seq_Features)
    X_test_seq_Features = array(X_test_seq_Features).reshape(X_test_seq_len,59,20,1)
    X_test_stru = data.loc[:,'ss0':'psa176']
    X_test_stru_Features = X_test_stru
    X_test_stru_len = len(X_test_stru_Features)
    X_test_stru_Features = array(X_test_stru_Features).reshape(X_test_stru_len,59,7,1)
    X_test_1D = data.loc[:,'fea1':'fea28']
    X_test_1D = array(X_test_1D).reshape(-1,X_test_1D.shape[1],1)
    X_test_seq = X_test_seq_Features
    X_test_stru = X_test_stru_Features
    X_test_1D = X_test_1D    
    m =load_model(model_path) 
    predict = m.predict([X_test_seq, X_test_stru, X_test_1D])
    pre = predict.tolist()
    pred = []
    for i in range(len(pre)):
        temp=pre[i][1]
        pred.append(temp)
    predict2 = np.array(pred)
    df = DataFrame({'Name': seq_name, 'Sequence': seq, 'Pos': pos,'WtAA': wtAA,'MutAA': mutAA,'Score': predict2.flatten()},
                   index=range(len(seq_name))) 
    csv_path = data_path + jobid + '.csv'
    df.to_csv(csv_path, index=False, encoding='gbk',float_format='%.3f')

 
def mkdir(path):
    import os
    path=path.strip() 
    path=path.rstrip("\\") 
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        return path
    else:
        return path

if __name__ == '__main__':
    import os
    import sys    
    jobid = sys.argv[1]
    email = sys.argv[2]
    if jobid.split() != "":
        data_path = rootPath + "/PredictDataOFUsers/"+ jobid + "/" 
        mkdir(data_path) 
        temp_path = rootPath + "/PredictDataOFUsers/" + jobid + ".txt" 
        os.system('cp -rf '+ temp_path+' '+  data_path)
        model_path = rootPath + "/model/" + "PRE_59_27_1-28D-FFMS-Resnet.h5"
        main(data_path=data_path, model_path=model_path, jobid=jobid)
        sendemail.sendmail(jobid=jobid, email=email.strip(), job_path=data_path) 