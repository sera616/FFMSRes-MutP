# coding: utf-8

# ---curPath:  /data1/commonuser/toolbar/webapps/gefang/ffms/code
# ---rootPath:  /data1/commonuser/toolbar/webapps/gefang/ffms
# ---data_path:  /data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/jobid/
# ---model_path:  /data1/commonuser/toolbar/webapps/gefang/ffms/code/model/PRE_59_27_1-28D-FFMS-Resnet.h5


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 警告过滤器
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



## read_text 读取用户提交的突变序列信息，分隔开并返回，供后面使用
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



## main 主函数
def main(data_path, model_path, jobid):
    
    seq_name, seq, pos, wtAA, mutAA = read_text(data_path,jobid)
    feature_generate(data_path,jobid,seq_name, seq, pos, wtAA, mutAA,)#使用pssm, pss, psa, pdo四种工具预测特征文件
    feature_extract(data_path,jobid)## 根据预测的特征文件，提取并保存为特征csv文件
    
    ## 读取特征csv文件,并reshape为model需要的input样式
    ## 导入突变的特征文件，使用保存的深度模型进行预测 ##
    import pandas as pd
    import numpy
    from numpy import array
    
    file_openpath = data_path + "PSPP1D28.csv"
    data = pd.read_csv(file_openpath)
    y_test = data['Label']

    '''
    1.先将特征进行reshape处理
    '''
    X_test_seq = data.loc[:,'pssm0':'pssm1179']
    print(X_test_seq.shape)
    X_test_seq_Features = X_test_seq
    X_test_seq_len = len(X_test_seq_Features)
    print(X_test_seq_Features.shape)
    X_test_seq_Features = array(X_test_seq_Features).reshape(X_test_seq_len,59,20,1)

    X_test_stru = data.loc[:,'ss0':'psa176']
    print(X_test_stru.shape)
    X_test_stru_Features = X_test_stru
    X_test_stru_len = len(X_test_stru_Features)
    print(X_test_stru_Features.shape)
    X_test_stru_Features = array(X_test_stru_Features).reshape(X_test_stru_len,59,7,1)

    X_test_1D = data.loc[:,'fea1':'fea28']
    X_test_1D = array(X_test_1D).reshape(-1,X_test_1D.shape[1],1)
    print(X_test_1D.shape,type(X_test_1D))

    X_test_seq = X_test_seq_Features
    X_test_stru = X_test_stru_Features
    X_test_1D = X_test_1D

    '''
        2.使用已保存的深度模型，进行预测
        PS：保存的模型包括 模型结构+权重
    '''
    import numpy as np
    from pandas import DataFrame
    from keras.models import load_model
    
    m =load_model(model_path)  # 加载训练好的模型
    predict = m.predict([X_test_seq, X_test_stru, X_test_1D])
    print('--predict--',predict) ##[[0.48025528 0.5197447 ]]
    ## 将numpy.ndarray转化为list，并只保留第1列数据（非第0列）
    pre = predict.tolist()
    pred = []
    for i in range(len(pre)):
        temp=pre[i][1]
        pred.append(temp)

    print('--pred--',pred)# [0.8639336228370667, 0.8639336228370667, 0.8639336228370667, 0.8639336228370667, 0.8639336228370667]
    predict2 = np.array(pred)
    print(predict2.flatten())
    print('--predict.flatten()--',predict2.flatten())

    print("Name:",seq_name, type(seq_name))
    print('Sequence:', seq)
    print("Pos",pos)
    print("wtAA", wtAA)
    print("mutAA", mutAA)
    print("'Score': ",predict2.flatten())
    print("index",len(wtAA))

    df = DataFrame({'Name': seq_name, 'Sequence': seq, 'Pos': pos,'WtAA': wtAA,'MutAA': mutAA,'Score': predict2.flatten()},
                   index=range(len(seq_name))) 

    csv_path = data_path + jobid + '.csv'
    df.to_csv(csv_path, index=False, encoding='gbk',float_format='%.3f')

    
'''
mkdir:新建文件夹，path为文件夹所在目录
此处：为每个jobid新建一个文件，如"/PredictDataOFUsers/20210422232231/"

'''    
def mkdir(path):
    # 引入模块
    import os
    path=path.strip()   # 去除首位空格
    path=path.rstrip("\\")    # 去除尾部 \ 符号
    # 判断路径是否存在: # 存在(True); # 不存在(False)
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录 ----># 创建目录操作函数
        os.makedirs(path) 
#         print (path+' 创建成功')
        return path
    else:
        # 如果目录存在则不创建，并提示目录已存在
#         print( path+' 目录已存在')
        return path
    
    


if __name__ == '__main__':
    import os
    import sys
    
    jobid = sys.argv[1]
#     model_choose = sys.argv[2]
    email = sys.argv[2]

    # jobid = '20210811105112'
    # email = 'gfang0616@njust.edu.cn'

    if jobid.split() != "":
        # 数据路径
#         data_path = rootPath + "/PredictDataOFUsers/"
        data_path = rootPath + "/PredictDataOFUsers/"+ jobid + "/" ## 每个jobid一个文件夹
        mkdir(data_path) #创建 data_path文件夹
        
        ## 将jobid.txt文件复制到data_path下面 （因为data_path是唯一的，所以可以区分开不同的job）
        temp_path = rootPath + "/PredictDataOFUsers/" + jobid + ".txt" 
        os.system('cp -rf '+ temp_path+' '+  data_path)## 将生成的jobid.txt文件复制到data_path文件夹下面
        print('---cp jobid.txt文件到/PredictDataOFUsers/jobid下成功---')
        
        # 模型路径
        model_path = rootPath + "/model/" + "PRE_59_27_1-28D-FFMS-Resnet.h5"
#         batch_size = 30
        # 模型预测模块
        main(data_path=data_path, model_path=model_path, jobid=jobid)
        # 邮件模块
        sendemail.sendmail(jobid=jobid, email=email.strip(), job_path=data_path) ##没问题





# In[31]:


# rootPath = "/data1/commonuser/toolbar/webapps/gefang/ffms"
# jobid = '20210810164355'
# data_path = rootPath + "/PredictDataOFUsers/"+ jobid + "/"
# data_path


# In[27]:


# ## list

# from pandas import DataFrame
# seq_name=['P10230','P12052']
# seq=['AHJHIVAUFBJWGALJGNIO','HIHUIHFGAHGOIHOI']
# pos=[10,18]
# wtAA=['A','F']
# mutAA=['E','D']
# Score= [0.86393362,0.5000000]
# # SEQ_name = [seq_name]

# df = DataFrame({'Name': seq_name, 'Sequence': seq, 'Pos': pos,'wtAA': wtAA,'mutAA': mutAA,'Score': Score}, index=range(len(seq_name)))

# csv_path =  './temp.csv'
# df.to_csv(csv_path, index=False, encoding='gbk',float_format='%.3f')


# In[28]:


# from pandas import DataFrame

# seq_name=['P10230','P12052']
# seq=['AHJHIVAUFBJWGALJGNIO','HIHUIHFGAHGOIHOI']
# pos=[10,18]
# wtAA=['A','F']
# mutAA=['E','D']
# Score= [0.86393362,0.5000000]
# # SEQ_name = [seq_name]

# df = DataFrame({'Name': seq_name, 'Sequence': seq, 'Pos': pos,'wtAA': wtAA,'mutAA': mutAA,'Score': Score})

# csv_path =  './temp.csv'
# df.to_csv(csv_path,  encoding='gbk',float_format='%.3f')

