
# coding: utf-8

# In[1]:



# coding: utf-8

# In[ ]:


# ---curPath:  /data1/commonuser/toolbar/webapps/gefang/ffms/code
# ---rootPath:  /data1/commonuser/toolbar/webapps/gefang/ffms
# ---data_path:  /data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/
# ---model_path:  /data1/commonuser/toolbar/webapps/gefang/ffms/code/model/PRE_59_27_1-28D-FFMS-Resnet.h5

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# # 警告过滤器
# import warnings
# warnings.filterwarnings("ignore")
# import sys

# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)


'''
get_filename(result_filepath):
            是获取软件得到的特征文件夹result_filepath下的文件名称，
            为后面根据文件名找文件做准备
            return (filename):返回该文件夹下所有文件的名称
'''
def get_filename(result_filepath):
    ## -------------- 1.获取路径下文件的名字---------------------------###
    import math
    import os  
    path = result_filepath
    files= os.listdir(path) ## files是当前路径下所有文件的名字+后缀
    filename=[]
    for i in range(len(files)):
        tem=files[i].split('.')[0]
        filename.append(tem)
    return (filename)


'''
get_POS_Name_Label(original_filepath,original_sheet_name)：
        original_filepath：是与特征文件夹对应的原始样本数据路径
        original_sheet_name：打开excel中的相应的工作表  
        return(POS,ID,label)：返回原始文件的POS,ID,label，最主要的是三者的index是完全一致的
'''
def get_POS_Name_Label(original_filepath,original_sheet_name):    
    ## -------------- 2.记录突变的名称，位置，以及标签信息--------------###
    import pandas as pd
    from pandas import DataFrame
    from AA3T1 import mut_split   ##----之前写的py程序，包括3AA1，和序列突变位点替换 两个函数---##

    pd = pd.read_excel(original_filepath, sheet_name=original_sheet_name)

    Variation = pd['Variation'].tolist()
    Name = pd['Name'].tolist()
    Label = pd['Label'].tolist()


    POS = []  ##---记录突变的位置，为提取特征做准备，特别注意，该位置为Name是严格对应关系
    ID = []  ##----记录与突变位置对应
    label =[]
    for i in range(len(Variation)):
        qian,pos,hou = mut_split(Variation[i])
        POS.append(pos)
        ID.append(Name[i])
        label.append(Label[i])
        
    return(POS,ID,label)



##  对于PSSM，有20个值；对于pdo仅有1个值。下面的程序需要改成更一般性的-----
'''
QH_vetor(flines,num,length)：提取并计算突变前后的vector，并返回
    flines：特征文件夹下的所有文件
    num ：突变的位置
    length :提取突变微环境的大小，这里的length是突变前（后）的微环境长度
     return(Q_vector,H_vector) ：返回计算权重后的前，后特征vector
'''

## -----------3.每个特征值，提取并计算突变前后的vector，合并成一行，并返回-----------###
##-------------提取并计算突变前后的vector，并返回------------##
def QH_vetor(flines,num,length,hang_size): 
    import numpy as np
    
    # 若突变位点前面没有29个残基，则从下标 （num-length=-20）
    if((num-length)<0):
        qian_value = flines[0:num]
    else:
        qian_value = flines[num+1:num+length+1]
    print('--qian_value--',qian_value)
        
    ## 若突变位点后面没有29个残基，则取到-1的位置，即取到倒数第一个;如果溢出了，则为空
    if((num+1)<len(flines)):
        hou_value = flines[num+1:num+length+1]
    else:
        hou_value = []
    print('--hou_value--',hou_value)


    ## -- 前  
    qian_join = []
    for i in range(len(qian_value)):
        tempQ =qian_value[i].strip().split('    ')## 将类似的'0.691', '0.3', '0.007'的string类型值转化为float类型
        to_floatQ = map(float,tempQ)
        qian_join.extend(to_floatQ)
    print('--qian_join--',qian_join)
    
    ## -- 后      
    hou_join =[]
    for i in range(len(hou_value)):
        tempH =hou_value[i].strip().split('    ')
        to_floatH = map(float,tempH)
        hou_join.extend(to_floatH)
    print('--hou_join--',hou_join)
    
    return(qian_join,hou_join) 



'''
QHp_list3to1(Q_vector,pos_value,H_vector):将Q_vector,pos_value，H_vector转化为1*9（行）的形式
    Q_vector:突变前特征向量
    pos_value:突变点特征向量
    H_vector:突变后特征向量
    return (feature)：返回一个样本的特征。对于SS，feature是1*9（行）的形式
'''
    
## -----------4.将Q_vector,pos_value，H_vector转化为1*9（行）的形式----------###
def QHp_list3to1(Q_vector,pos_value,H_vector,MicroEn_length,hang_size):
    ##-------将Q_vector,pos_value，H_vector保存到excel中---------------##
    ## 1----将Q_vector,pos_value，H_vector转化为1*9（行）的形式
    
    feature = []
    qian = []
    center = []
    hou = []
    qian_value = []
    hou_value = []
    
    for hang in range(len(Q_vector)):
        tp = Q_vector[hang]
        qian.append(tp)
    ## 将空缺位置补0 --- 若突变位点前面没有29个残基，则从下标 （num-length=-20）
    # if (len(qian)<MicroEn_length*hang_size):
        qian_temp = [0 for i in range((MicroEn_length*hang_size)-len(qian))]
        qian_value = qian_temp + qian ##两个list合并成一个list
    print('-----QHp_list3to1------')
    print('--qian_value--',qian_value)

    for hang in range(len(pos_value)):
        tp = pos_value[hang]
        center.append(tp)
    print('--center--',center)
        
    for hang in range(len(H_vector)):
        tp = H_vector[hang]
        hou.append(tp)
    ## 将空缺位置补0 --- 若突变位点后面没有29个残基，则取到-1的位置，即取到倒数第一个;如果溢出了，则为空
        hou_temp = [0 for i in range((MicroEn_length*hang_size)-len(hou))]
        hou_value = hou+ hou_temp##两个list合并成一个list
    print('--hou_value--',hou_value)
        
    feature = qian_value + center+ hou_value
    print('--feature--',feature)
    return (feature)


'''
QHp_list3to1_PSA(Q_vector,pos_value,H_vector):将Q_vector,pos_value，H_vector转化为1*9（行）的形式
    Q_vector:突变前特征向量
    pos_value:突变点特征向量
    H_vector:突变后特征向量
    return (feature)：返回一个样本的特征。对于SS，feature是1*9（行）的形式
'''
    
## -----------4.将Q_vector,pos_value，H_vector转化为1*9（行）的形式----------###
def QHp_list3to1_PSA(Q_vector,pos_value,H_vector,MicroEn_length,hang_size):
    ##-------将Q_vector,pos_value，H_vector保存到excel中---------------##
    ## 1----将Q_vector,pos_value，H_vector转化为1*9（行）的形式
    
    feature = []
    qian = []
    center = []
    hou = []
    qian_value = []
    hou_value = []
    
    for hang in range(len(Q_vector)):
        tp = Q_vector[hang]
        qian.append(tp)
        
    ## 将空缺位置补0 --- 若突变位点前面没有29个残基，则从下标 （num-length=-20）
        qian_temp = [0 for i in range((MicroEn_length*hang_size)-len(qian))]
        qian_value = qian_temp + qian ##两个list合并成一个list

    for hang in range(len(pos_value)):
        tp = pos_value[hang]
        center.append(tp)
        
    for hang in range(len(H_vector)):
        tp = H_vector[hang]
        hou.append(tp)
    ## 将空缺位置补0 --- 若突变位点后面没有29个残基，则取到-1的位置，即取到倒数第一个;如果溢出了，则为空
        hou_temp = [0 for i in range((MicroEn_length*hang_size)-len(hou))]
        hou_value = hou+ hou_temp##两个list合并成一个list
        
    feature = qian_value + center+ hou_value
    return (feature)


'''
get_fea_IDN_LabelN(result_filepath,filename,POS,ID,label,MicroEn_length)：
    result_filepath：使用软件所得到的特征文件夹路径
    filename：该路径下所有文件的名称
    
    POS：原始文件中的突变位置POS
    ID：原始文件中的突变名称ID
    label：原始文件中的突变标签label。上面三者的index完全对应
    
    MicroEn_length：需要提取微环境的长度。以突变点为中心，前后的长度
    return(feature,IDName,Labelname)：返回所有样本的feature,每一行是一个样本的特征。
                                        并且对应的IDName和Labelname也返回，便于后面的文件保存。
    
'''

def get_fea_IDN_LabelN(result_filepath,filename,POS,ID,label,MicroEn_length,hang_size): 
    import os
    ##-------------5. 提取所有样本的特征，以及与特征对应的name和标签-----##
    feature = []  ##用于保存全部样本的特征
    IDName = [] ##用于保存全部样本的名称，与所提取的特征相对应的
    Labelname = []  ##用于保存全部样本的标签，与所提取的特征相对应的
    
    ##--传进来的参数-----##
    POS = POS
    ID = ID 
    label = label
    hang_size = hang_size ##记录每一行的特征个数，如ss为3; pssm为20.
    
    for j in range(len(filename)):
        xiabiao=ID.index(filename[j])##filename[j]在ID list中的下标，决定了突变position的值
        Position= POS[xiabiao]##突变position的值
        
        path = result_filepath
        files= os.listdir(path) ## files是当前路径下所有文件的名字+后缀
        f = open(path+"/"+files[j]) #打开文件 ##打开files[j] 文件
        flines=f.readlines() ## 读第j个文件

        num=int(Position)-1 ## 突变下标 = 突变位置-1
        IDname=ID[xiabiao]
        labelname=label[xiabiao]
        
        IDName.append(IDname)
        Labelname.append(labelname)


        pos_value = []  ## 将字符形式['0.616', '0.324', '0.145']，改写成float类型，便于后面的保存
        pos_va = flines[num].strip().split('    ')
        for p in range(len(pos_va)):
            t = float(pos_va[p])
            pos_value.append(t)
            
        Q_vector, H_vector = QH_vetor(flines,num,MicroEn_length,hang_size)##---获取突变点前后的vector       
        
        ##---将每个样本的特征，先存放在temp_feature中
        temp_feature =  QHp_list3to1(Q_vector,pos_value,H_vector,MicroEn_length,hang_size)

        feature.append(temp_feature)
        f.close()
        
    return(feature,IDName,Labelname)


'''
get_fea_IDN_LabelN(result_filepath,filename,POS,ID,label,MicroEn_length)：
    result_filepath：使用软件所得到的特征文件夹路径
    filename：该路径下所有文件的名称
    
    POS：原始文件中的突变位置POS
    ID：原始文件中的突变名称ID
    label：原始文件中的突变标签label。上面三者的index完全对应
    
    MicroEn_length：需要提取微环境的长度。以突变点为中心，前后的长度
    return(feature,IDName,Labelname)：返回所有样本的feature,每一行是一个样本的特征。
                                        并且对应的IDName和Labelname也返回，便于后面的文件保存。
    
'''

def get_fea_IDN_LabelN_PSA(result_filepath,filename,POS,ID,label,MicroEn_length,hang_size): 
    import os
    ##-------------5. 提取所有样本的特征，以及与特征对应的name和标签-----##
    feature = []  ##用于保存全部样本的特征
    IDName = [] ##用于保存全部样本的名称，与所提取的特征相对应的
    Labelname = []  ##用于保存全部样本的标签，与所提取的特征相对应的
    
    ##--传进来的参数-----##
    POS = POS
    ID = ID 
    label = label
    hang_size = hang_size ##记录每一行的特征个数，如ss为3; pssm为20.
    
    for j in range(len(filename)):
        xiabiao=ID.index(filename[j])##filename[j]在ID list中的下标，决定了突变position的值
        Position= POS[xiabiao]##突变position的值
        
        path = result_filepath
        files= os.listdir(path) ## files是当前路径下所有文件的名字+后缀
        f = open(path+"/"+files[j]) #打开文件 ##打开files[j] 文件
        flines=f.readlines() ## 读第j个文件
        
        num=int(Position)-1 ## 突变下标 = 突变位置-1
        IDname=ID[xiabiao]
        labelname=label[xiabiao]
        
        IDName.append(IDname)
        Labelname.append(labelname)


        pos_value = []  ## 将字符形式['0.616', '0.324', '0.145']，改写成float类型，便于后面的保存
        pos_va = flines[num].strip().split('    ')
        for p in range(len(pos_va)):
            t = float(pos_va[p])
            pos_value.append(t)
            
        Q_vector, H_vector = QH_vetor(flines,num,MicroEn_length,hang_size)##---获取突变点前后的vector       
        
        ##---将每个样本的特征，先存放在temp_feature中
        temp_feature =  QHp_list3to1_PSA(Q_vector,pos_value,H_vector,MicroEn_length,hang_size)           

        feature.append(temp_feature)
        f.close()
        
    return(feature,IDName,Labelname)


'''
feature_name(string,length):
    string: 保存特征文件时，每列的列名称
    length: 列名称的长度
    return (feature_name_list):返回列名称list
'''
## ------ 特征名字列表---------#
def feature_name(string,length):
    feature_name_list = []
    for i in range(0,length):
        tp = ''
        temp =string + str(i)
        feature_name_list.append(temp)
    return (feature_name_list)


'''
save_file(fea_res_fpath,column_name,column_na_length):
    fea_res_fpath:保存文件的位置
    column_name：列名称
    column_na_length：列名称的长度
    return('文件保存成功！')
'''
def save_file(fea_res_fpath,column_name,column_na_length, feature, IDName,Labelname ):   
    ## -------------6. 获取feature，IDName，Labelname --------------###
    import pandas as pd

    feature_name_list = []
    feature_name_list = feature_name (column_name,column_na_length)

    ##-- 将feature，IDName，以及Labelname转化为dataframe，用于后面的文件保存
    fea_pd = pd.DataFrame(feature[:])
    fea_ID = pd.DataFrame(IDName[:])
    fea_Labelname = pd.DataFrame(Labelname[:])


    fea_pd.columns = [feature_name_list] ## 添加列名称列的名称
    fea_pd.insert(0,'Name',fea_ID )     #插入一列
    fea_pd.insert(1,'Label',fea_Labelname)     #插入一列

    # 保存到本地excel
    
    fea_pd.to_csv(fea_res_fpath, index=True)

    return('文件保存成功！')



# In[3]:


# ## read_text 读取用户提交的突变序列信息，分隔开并返回，供后面使用
# def read_text(data_path, jobid):
#     seq_path = data_path + jobid + ".txt"
#     print(seq_path)
#     with open(seq_path, "r") as f:
#         lines=f.readlines()
#         head = []

#         seq_name =[]
#         seq = []
#         pos=[]
#         wtAA = []
#         mutAA = []

#         for i in range(len(lines)):
#             if (lines[i][0]=='>'):
#                 temp = lines[i].strip()
#                 head.append(temp)
#             else:
#                 seq.append(lines[i].strip())

#         for j in range(len(head)):
#             hp = head[j].split('_')[0]
#             seq_name.append(hp[1:])

#             mp = head[j].split('_')[1]                  
#             wtAA.append(mp[0])
#             pos.append(mp[1:-1])
#             mutAA.append(mp[-1])
        
#     return (seq_name, seq, pos, wtAA, mutAA)

# data_path="./"
# jobid = "20210811105112"
# seq_name, seq, pos, wtAA, mutAA = read_text(data_path)
# print(seq_name)


# In[ ]:



'''
SS_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    根据预测软件获得的特征文件，提取突变位点为中心59AA范围内的特征信息。
    保存处理后的特征文件到/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/SS/SS_59.csv
    
    return 0
'''
def SS_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    filename = seq_name
    POS = pos
    ID = seq_name
#     label = [-1]
    label = [-1 for i in range (len(seq_name))] ## 设置与seq_name长度相同的label----都为 -1
 
    file_openpath = data_path + "SS/"
    feature,IDName,Labelname = get_fea_IDN_LabelN(file_path,filename,POS,ID,label,29,3) #(59-1)/2=29; 其中3为ss每一行的特征个数
    PSS_fea = feature
    print('---PSS_fea---',PSS_fea,len(PSS_fea))

    file_savepath = data_path + "SS/SS_59.csv"
    save_file(file_savepath,'ss',177)  # (29+29+1)*3=177
    
    return 0


# In[ ]:




'''
SS_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    根据预测软件获得的特征文件，提取突变位点为中心59AA范围内的特征信息。
    保存处理后的特征文件到/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/SS/SS_59.csv
    
    return 0
'''
def SS_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    filename = seq_name
    POS = pos
    ID = seq_name
#     label = [-1]
    label = [-1 for i in range (len(seq_name))] ## 设置与seq_name长度相同的label----都为 -1

    file_openpath = data_path + "SS/"
    feature,IDName,Labelname = get_fea_IDN_LabelN(file_openpath,filename,POS,ID,label,29,3) #(59-1)/2=29; 其中3为ss每一行的特征个数
#     feature,IDName,Labelname = get_fea_IDN_LabelN("./SS/",filename,POS,ID,label,29,3) #(59-1)/2=29; 其中3为ss每一行的特征个数
    PSS_fea = feature
    print('---PSS_fea---',PSS_fea,len(PSS_fea))

#     save_file("./SS/SS_59.csv",'ss',177)  # (29+29+1)*3=177
    file_savepath = data_path + "SS/SS_59.csv"
    save_file(file_savepath,'ss',177, PSS_fea, ID,label) # (29+29+1)*3=177
    
    return 0


'''
PSSM_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    根据预测软件获得的特征文件，提取突变位点为中心59AA范围内的特征信息。
    保存处理后的特征文件到/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/PSSM/PSSM_59.csv
    
    return 0
'''
def PSSM_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    filename = seq_name
    POS = pos
    ID = seq_name
#     label = [-1]
    label = [-1 for i in range (len(seq_name))] ## 设置与seq_name长度相同的label----都为 -1

    file_openpath = data_path + "PSSM/"
    feature,IDName,Labelname = get_fea_IDN_LabelN(file_openpath,filename,POS,ID,label,29,20) #(59-1)/2=29; 其中3为ss每一行的特征个数
    PSSM_fea = feature
#     print('---PSSM_fea---',PSSM_fea,len(PSSM_fea))

    file_savepath = data_path + "PSSM/PSSM_59.csv"
    save_file(file_savepath,'pssm',1180, PSSM_fea, ID,label) # (29+29+1)*20=1180
    
    return 0



'''
PDO_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    根据预测软件获得的特征文件，提取突变位点为中心59AA范围内的特征信息。
    保存处理后的特征文件到/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/PDO/PDO_59.csv
    
    return 0
'''
def PDO_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    filename = seq_name
    POS = pos
    ID = seq_name
#     label = [-1]
    label = [-1 for i in range (len(seq_name))] ## 设置与seq_name长度相同的label----都为 -1

    file_openpath = data_path + "PDO/"
    feature,IDName,Labelname = get_fea_IDN_LabelN(file_openpath,filename,POS,ID,label,29,1) #(59-1)/2=29; 其中3为ss每一行的特征个数
    PDO_fea = feature
#     print('---PDO_fea---',PDO_fea,len(PDO_fea))

    file_savepath = data_path + "PDO/PDO_59.csv"
    save_file(file_savepath,'pdo',59, PDO_fea, ID,label) # (29+29+1)*1=59
        


'''
PSA_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    根据预测软件获得的特征文件，提取突变位点为中心59AA范围内的特征信息。
    保存处理后的特征文件到/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/PSA/PSA_59.csv
    
    return 0
'''
'''
由于PAS使用的截取501AA后的序列。
因此，使用的函数：
    get_fea_IDN_LabelN_PSA（）
    QHp_list3to1_PSA（）
    
与PSSM,PSS.PDO不同
'''

def PSA_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    filename = seq_name
    seq = seq #原先的protein sequence
    pos = pos #原先的protein mutation site
#     PSA_POS = [250] ## PSA_POS是截取之后的突变位点定位
    PSA_POS = [250 for i in range(len(seq_name))]

    ID = seq_name
#     label = [-1]
    label = [-1 for i in range (len(seq_name))] ## 设置与seq_name长度相同的label----都为 -1
    
    ##从得到的特征文件夹中，获得相应的文件，并提取突变点的微环境特征，IDname, Labelname
    file_openpath = data_path + "PSA/"
    feature,IDName,Labelname = get_fea_IDN_LabelN_PSA(file_openpath,filename,PSA_POS,ID,label,29,3) #(59-1)/2=29; 其中3为ss每一行的特征个数
    PSA_fea = feature

    ## 保存得到的特征文件,第二个参数是特征的列名称，第三个参数是列名称的长度
    file_savepath = data_path + "PSA/PSA_59.csv"
    save_file(file_savepath,'psa',177, PSA_fea, ID,label) # (29+29+1)*3=177
    
    return 0



## 1D-28 特征的提取

import sys,os,string
from Bio import SeqIO
import urllib.request

amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','R','Q','S','T','V','W','Y']

dict_31 = {
    "GLY": "G","ALA": "A","VAL": "V","ILE": "I","LEU": "L","PRO": "P",
    "SER": "S","THR": "T","CYS": "C","MET": "M","ASP": "D","ASN": "N",
    "GLU": "E","GLN": "Q","LYS": "K","ARG": "R","HIS": "H","PHE": "F",
    "TYR": "Y","TRP": "W"}

dict_13={}
for key in dict_31.keys():
    dict_13[dict_31[key]]=key

#提取 物理化学属性特征
def parse_physico_chemical_property_aaindex(fn):
    """
    Parses file with physico chemical properties.
    See example http://www.genome.jp/dbget-bin/www_bget?aaindex:FASG890101
    :param filename:
    :return: returns dict key = one letter name of an amino acid
                          value = physico chemical property

    # KUHL950101 Hydrophilicity scale (Kuhn et al., 1995)
    # MITS020101 Amphiphilicity index (Mitaku et al., 2002)
    # ZIMJ680102 Bulkiness (Zimmerman et al., 1968)
    # GRAR740102 Polarity (Grantham, 1974)
    # CHAM820101 Polarizability parameter (Charton-Charton, 1982)
    # ZIMJ680104 Isoelectric point (Zimmerman et al., 1968)
    # CHOC760101 Residue accessible surface area in tripeptide (Chothia, 1976)
    # FAUJ880109 Number of hydrogen bond donors (Fauchere et al., 1988)
    # KLEP840101 Net charge (Klein et al., 1984)
    # LEVM760105 Radius of gyration of side chain (Levitt, 1976)
    # CEDJ970103 Composition of amino acids in membrane proteins (percent) (Cedano et al.,  1997)
    # TAKK010101 Side-chain contribution to protein stability (kJ/mol) (Takano-Yutani, 2001)

    """
    # ？？？fn是什么？
    f = open(fn, 'r')
    ls = f.readlines()
    f.close()
    ls_aa = ls[-3].rstrip().split()[1:] # pair of aa
    ls_p1 = ls[-2].rstrip().split() # value for 1st aa
    ls_p2 = ls[-1].rstrip().split() # value for 2nd aa

    if len(ls_aa)!=len(ls_p1) or len(ls_aa)!=len(ls_p2):
        raise ValueError("Bad Parsing")

    prop={}
    for i in range(len(ls_aa)):
        aa1=ls_aa[i].split('/')[0]
        aa2=ls_aa[i].split('/')[1]
        p1 = float(ls_p1[i])
        p2 = float(ls_p2[i])

        prop[aa1]=p1
        prop[aa2]=p2

    return prop


#提取替代打分矩阵特征
def parse_substitution_matrix_aaindex(fn):

    '''
    # NGPC000101 Substitution matrix (PHAT) built from hydrophobic and transmembrane regions of the Blocks database (Ng et al., 2000)
    # MUET010101 Non-symmetric substitution matrix (SLIM) for detection of homologous transmembrane proteins (Mueller et al., 2001)
    # HENS920102 BLOSUM62 substitution matrix (Henikoff-Henikoff, 1992)
    '''

    f=open(fn, 'r')
    ls =f.readlines()
    f.close()

    substitution_matrix = {}

    i_start=0
    for i, s in enumerate(ls):
        if s[0]!='M':continue
        i_start=i
        break

    rows = ls[i_start].rstrip().split()[3].strip(',') # ARNDCQEGHILKMFPSTWYV
    cols = ls[i_start].rstrip().split()[6].strip(',') # ARNDCQEGHILKMFPSTWYV

    if rows!=cols:
        print (rows, cols)
        raise ValueError('Matrix must be squared')

    for i in range(0,len(rows)):
        data = ls[i_start + 1 + i].rstrip().split() #-5      -6       0       6      -7       1      12
        for j in range(0,i+1):
            pair_aa = rows[i]+cols[j]
            value = float(data[j])
            substitution_matrix[pair_aa]=value

    return substitution_matrix

#提取替换打分矩阵特征
def parse_substitution_matrix_slim(fn):

    f = open(fn, 'r')
    ls = f.readlines()[1:] # first line is a comment
    f.close()

    substitution_matrix = {}

    ls_aa = ls[0].rstrip().split('\t')[1:] # first el is 'aa/aa'
    for s in ls[1:]:
        data = s.rstrip().split('\t')
        aa1 = data[0]
        for i in range(0,len(ls_aa)):
           aa2 = ls_aa[i]
           value = float(data[i+1])
           substitution_matrix[aa1+aa2]=value

    return substitution_matrix

##将3字母AA改为1字母AA
def getAA(aa, code=1):
# this function convert amino acid into one letter upper case for code=1(e.g. A), or to three letter upper case (e.g. ALA) for code=3
    if code==1:
        if len(aa)==1 and aa.upper() in dict_13.keys(): return aa.upper()
        if len(aa)==3 and aa.upper() in dict_31.keys(): return dict_31[aa.upper()]
    elif code==3:
        if len(aa)==1 and aa.upper() in dict_13.keys(): return dict_13[aa.upper()]
        if len(aa)==3 and aa.upper() in dict_31.keys(): return aa.upper()

    print (aa)
    raise ValueError('Wrong aa code')
    exit(1)

    return 1

#获取物理化学属性
dir = "/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/resource/"
# def getPhysicoChemicalProperties(dir='./resource/', ext='.txt'):
def getPhysicoChemicalProperties(dir, ext='.txt'):
    # this function returns 6-tuple of tabulated properties for the amino acid
    properties = ['KUHL950101', 'MITS020101', 'ZIMJ680102', 'GRAR740102',
                  'CHAM820101', 'ZIMJ680104', 'CHOC760101', 'FAUJ880109',
                  'KLEP840101', 'LEVM760105', 'CEDJ970103', 'TAKK010101']


    physico_chemical_properties = []
    for prop in properties:
        fn = dir+prop+ext
        if not os.path.exists(fn):
            print (properties)
            print (dir, prop, ext, fn)
            raise ValueError('no such file')
        physico_chemical_properties.append(parse_physico_chemical_property_aaindex(fn))

    return physico_chemical_properties


#获取BlosumScore
def getBlosumScore(aa1, aa2):
    # this function returns substitution score from the Blosum matrix for aa1 to aa2 mutation

    global blosum62_matrix

    a1 = getAA(aa1)
    a2 = getAA(aa2)

    if a1+a2 in blosum62_matrix.keys():
        return blosum62_matrix[a1+a2]
    elif a2+a1 in blosum62_matrix.keys():
        return blosum62_matrix[a2+a1]
    else:
        print(a1, a2)
        raise ValueError('No corresponding keys in the Blosum matrix')

    return -1

#获取PhatScore
def getPhatScore(aa1, aa2):
    # this function returns substitution score from the PHAT matrix for aa1 to aa2 mutation

    global phat_matrix

    a1 = getAA(aa1)
    a2 = getAA(aa2)

    if a1+a2 in phat_matrix.keys():
        return phat_matrix[a1+a2]
    elif a2+a1 in phat_matrix.keys():
        return phat_matrix[a2+a1]
    else:
        print (a1, a2)
        raise ValueError('No corresponding keys in the PHAT matrix')

    return -1

#获取SlimScore
def getSlimScore(aa1, aa2):
    # this function returns substitution score from the SLIM matrix for aa1 to aa2 mutation

    global slim_matrix

    a1 = getAA(aa1)
    a2 = getAA(aa2)

    if a1+a2 in slim_matrix.keys():
        return slim_matrix[a1+a2]
    else:
        print (a1, a2)
        raise ValueError('No corresponding keys in the SLIM matrix')

    return -1

# #
# def getGeneralDescriptorsUniprot(acc):

#     link = "http://www.uniprot.org/uniprot/"+acc+".xml"
# #     handle = urllib.urlopen(link)
#     handle = urllib.request.urlopen(link)
#     record = SeqIO.read(handle, "uniprot-xml")
#     seq_length = len(record)

#     nTM=0
#     nHelix = 0
#     nStrand = 0
#     nTurn = 0
#     for f in record.features:
#         if f.type=='transmembrane region': nTM+=1
#         if f.type=='helix': nHelix+=1
#         if f.type=='strand': nStrand+=1
#         if f.type=='turn': nTurn+=1

#     # probably it is only for structure
#     if nHelix==0 and nStrand==0 and nTurn==0:
#         nHelix=-100
#         nStrand=-100
#         nTurn=-100

#     s_log=''
#     if nTM==0:
#         s_log="Warning! There is no transmembrane regions! %s" % (acc)


#     descriptor=[seq_length, nTM, nHelix, nStrand, nTurn]
#     return descriptor, s_log


def calculateSequenceBasedDescriptor(acc, aa_wt, aa_mut, pos):
    # this functions calculate features from the sequence-based-descriptors and combine them into the one vector

    global properties

    aa_wt = getAA(aa_wt)
    aa_mut = getAA(aa_mut)

    descriptor = []
    s_log=''
    
    for prop in properties: descriptor.append(prop[aa_wt])
    for prop in properties: descriptor.append(prop[aa_mut])
    descriptor.append(getBlosumScore(aa_wt, aa_mut))
    descriptor.append(getPhatScore(aa_wt,aa_mut))
    descriptor.append(getSlimScore(aa_wt,aa_mut))
    descriptor.append(getSlimScore(aa_mut,aa_wt))
#     features, message = getGeneralDescriptorsUniprot(acc)
#     s_log+=message
#     for feature in features: descriptor.append(feature)
    
    return descriptor, s_log


# if __name__=="__main__":

#     # temp_path:  /data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/
#     blosum62_matrix = parse_substitution_matrix_aaindex(temp_path + "resource/substitution_matrix_blosum62.txt")
#     phat_matrix = parse_substitution_matrix_aaindex(temp_path + "resource/substitution_matrix_phat.txt")
#     slim_matrix = parse_substitution_matrix_slim(temp_path + "resource/substitution_matrix_slim161.txt")
#     properties = getPhysicoChemicalProperties()

#     flag_check = False
#     if flag_check==True:
#         aa_wt = 'L'
#         aa_mut = 'W'
#         descriptor=calculateSequenceBasedDescriptor('P41145', aa_wt, aa_mut, 0)
#         print (descriptor)


# In[ ]:

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
    

'''
sequence_1D28(data_path,seq_name, seq, pos, wtAA, mutAA):
    提取序列的28个特征，并保存文件


'''
def sequence_1D28(data_path,seq_name, seq, pos, wtAA, mutAA):
    
    # data_path:  /data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/
    global properties
    global slim_matrix
    global phat_matrix
    global blosum62_matrix

    temp_path = "/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/"
    blosum62_matrix = parse_substitution_matrix_aaindex(temp_path + "resource/substitution_matrix_blosum62.txt")
    phat_matrix = parse_substitution_matrix_aaindex(temp_path + "resource/substitution_matrix_phat.txt")
    slim_matrix = parse_substitution_matrix_slim(temp_path + "resource/substitution_matrix_slim161.txt")

    dir = temp_path + "resource/"
    properties = getPhysicoChemicalProperties(dir=dir, ext='.txt')
    
    data_path = data_path
    seq_name = seq_name
    seq = seq
    pos = pos
    wtAA = wtAA
    mutAA = mutAA 



    sequence_fea = []
    for i in range(len(seq_name)):
        descriptor=calculateSequenceBasedDescriptor(seq_name[i], wtAA[i], mutAA[i], pos[i])
        #     print (descriptor,len(descriptor[0]))
        tempsequence_fea = descriptor[0]
        sequence_fea.append(tempsequence_fea)
    # print(sequence_fea)

    sequence_feaName = []
    for i in range(1,29,1):
        temp = 'fea'+str(i)
        sequence_feaName.append(temp)
        
    # print(len(sequence_feaName))
    ## -------------  ## 将特征进行标准化处理 + 特征保存到excel文件-----------###
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import minmax_scale

    X = np.array(sequence_fea)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)  
    # print(X_minmax)
    fea_pd = pd.DataFrame(X_minmax[:],) ##28行1列
    # fea_pd = fea_pd.T ##转置成1行28列

    fea_ID = pd.DataFrame(seq_name[:])
    label = [-1 for i in range (len(seq_name))] ## 设置与seq_name长度相同的label----都为 -1
    fea_Labelname = pd.DataFrame(label[:])

    fea_pd.columns = [sequence_feaName] ## 添加列名称列的名称
    fea_pd.insert(0,'Name',fea_ID )     #插入一列
    fea_pd.insert(1,'Label',fea_Labelname)     #插入一列
    
    ## 先创建一个文件夹，名为SeqFEA
    save_pssmpath = data_path+'SeqFEA/'
    mkdir(save_pssmpath)
    
    # 保存到本地excel
    file_savepath = data_path + "SeqFEA/sequencefea.csv"
    fea_pd.to_csv(file_savepath, index=True)
    print('sequencefea.csv 文件保存成功！')
    
    return 0
    


# In[ ]:

## read_text 读取用户提交的突变序列信息，分隔开并返回，供后面使用
def read_text(data_path,jobid):
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



## 根据预测的特征文件，提取并保存为特征csv文件
def feature_extract(data_path,jobid):
    seq_name,seq,pos,wtAA,mutAA = read_text(data_path,jobid) ## 读取TXT文件，返回的seq_name,seq,pos,wtAA,mutAA都是list格式
    
    SS_extract(data_path,seq_name, seq, pos, wtAA, mutAA) ## SS
    print("--SS_extract 执行完成--")
    PSSM_extract(data_path,seq_name, seq, pos, wtAA, mutAA)## PSSM
    print("--PSSM_extract 执行完成--")
    PDO_extract(data_path,seq_name, seq, pos, wtAA, mutAA)## PDO
    print("--PDO_extract 执行完成--")
    PSA_extract(data_path,seq_name, seq, pos, wtAA, mutAA)## PSA
    print("--PSA_extract 执行完成--")
    sequence_1D28(data_path,seq_name, seq, pos, wtAA, mutAA)## seqence_1D28
    print("--sequence_1D28 执行完成--")
    
    
    ## 将不同特征进行合并
    ## 把所有特征的空值，填充为0
    import pandas as pd 
    
    PSSM_path = data_path + "PSSM/PSSM_59.csv"
    df_PSSM = pd.read_csv(PSSM_path) 
    print(df_PSSM.shape)

    SS_path = data_path + "SS/SS_59.csv"
    df_SS = pd.read_csv(SS_path) 
    df_SS = df_SS.loc[:,'ss0':'ss176']## 把ID和Lable 列去掉
    print(df_SS.shape)

    PDO_path = data_path + "PDO/PDO_59.csv"
    df_PDO = pd.read_csv(PDO_path) 
    df_PDO = df_PDO.loc[:,'pdo0':'pdo58']## 把ID和Lable 列去掉
    print(df_PDO.shape)

    PSA_path = data_path + "PSA/PSA_59.csv"
    df_PSA= pd.read_csv(PSA_path)
    df_PSA = df_PSA.loc[:,'psa0':'psa176']## 把ID和Lable 列去掉
    print(df_PSA.shape)

    SeqFEA_path = data_path + "SeqFEA/sequencefea.csv"
    df_seq =  pd.read_csv(SeqFEA_path) 
    df_seq = df_seq.loc[:,'fea1':'fea28']## 把ID和Lable 列去掉
    print(df_seq.shape)

    df1 = pd.concat([df_PSSM,df_SS,df_PDO,df_PSA,df_seq],axis=1,ignore_index=False)  #将df2数据与df1合并
    
    file_savepath = data_path  + "PSPP1D28.csv"
    df1.to_csv(file_savepath,index=False) 
    print('PSPP1D28.csv 保存成功！')


# In[ ]:


# if __name__ == '__main__':
#     import os
#     import sys
    
# #     jobid = sys.argv[1]
# # #     model_choose = sys.argv[2]
# #     email = sys.argv[2]

#     jobid = '20210811105112'
#     email = 'gfang0616@njust.edu.cn'

#     if jobid.split() != "":
#         # 数据路径
# #         data_path = rootPath + "/PredictDataOFUsers/"
#         data_path = rootPath + "/PredictDataOFUsers/"+ jobid + "/" ## 每个jobid一个文件夹
#         mkdir(data_path) #创建 data_path文件夹
        
#         ## 将jobid.txt文件复制到data_path下面 （因为data_path是唯一的，所以可以区分开不同的job）
#         temp_path = rootPath + "/PredictDataOFUsers/" + jobid + ".txt" 
#         os.system('cp -rf '+ temp_path+' '+  data_path)## 将生成的jobid.txt文件复制到data_path文件夹下面
#         print('---cp jobid.txt文件到/PredictDataOFUsers/jobid下成功---')
        
#     # seq_name, seq, pos, wtAA, mutAA = read_text(data_path, jobid)
#     feature_extract(data_path,jobid)

