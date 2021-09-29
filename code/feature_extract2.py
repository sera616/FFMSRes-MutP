
# coding: utf-8

def get_filename(result_filepath):
    import math
    import os  
    path = result_filepath
    files= os.listdir(path)
    filename=[]
    for i in range(len(files)):
        tem=files[i].split('.')[0]
        filename.append(tem)
    return (filename)


def get_POS_Name_Label(original_filepath,original_sheet_name):    
    import pandas as pd
    from pandas import DataFrame
    from AA3T1 import mut_split 
    pd = pd.read_excel(original_filepath, sheet_name=original_sheet_name)
    Variation = pd['Variation'].tolist()
    Name = pd['Name'].tolist()
    Label = pd['Label'].tolist()
    POS = [] 
    ID = []
    label =[]
    for i in range(len(Variation)):
        qian,pos,hou = mut_split(Variation[i])
        POS.append(pos)
        ID.append(Name[i])
        label.append(Label[i])
        
    return(POS,ID,label)


def QH_vetor(flines,num,length,hang_size): 
    import numpy as np
    if((num-length)<0):
        qian_value = flines[0:num]
    else:
        qian_value = flines[num+1:num+length+1]
    print('--qian_value--',qian_value)
    if((num+1)<len(flines)):
        hou_value = flines[num+1:num+length+1]
    else:
        hou_value = []
    print('--hou_value--',hou_value)
    qian_join = []
    for i in range(len(qian_value)):
        tempQ =qian_value[i].strip().split('    ')
        to_floatQ = map(float,tempQ)
        qian_join.extend(to_floatQ)
    print('--qian_join--',qian_join)  
    hou_join =[]
    for i in range(len(hou_value)):
        tempH =hou_value[i].strip().split('    ')
        to_floatH = map(float,tempH)
        hou_join.extend(to_floatH)
    print('--hou_join--',hou_join)
    
    return(qian_join,hou_join) 

def QHp_list3to1(Q_vector,pos_value,H_vector,MicroEn_length,hang_size):   
    feature = []
    qian = []
    center = []
    hou = []
    qian_value = []
    hou_value = []    
    for hang in range(len(Q_vector)):
        tp = Q_vector[hang]
        qian.append(tp)
        qian_temp = [0 for i in range((MicroEn_length*hang_size)-len(qian))]
        qian_value = qian_temp + qian 
    print('-----QHp_list3to1------')
    print('--qian_value--',qian_value)
    for hang in range(len(pos_value)):
        tp = pos_value[hang]
        center.append(tp)
    print('--center--',center)        
    for hang in range(len(H_vector)):
        tp = H_vector[hang]
        hou.append(tp)
        hou_temp = [0 for i in range((MicroEn_length*hang_size)-len(hou))]
        hou_value = hou+ hou_temp
    print('--hou_value--',hou_value)        
    feature = qian_value + center+ hou_value
    print('--feature--',feature)
    return (feature)

def QHp_list3to1_PSA(Q_vector,pos_value,H_vector,MicroEn_length,hang_size):    
    feature = []
    qian = []
    center = []
    hou = []
    qian_value = []
    hou_value = []    
    for hang in range(len(Q_vector)):
        tp = Q_vector[hang]
        qian.append(tp)
        qian_temp = [0 for i in range((MicroEn_length*hang_size)-len(qian))]
        qian_value = qian_temp + qian
    for hang in range(len(pos_value)):
        tp = pos_value[hang]
        center.append(tp)        
    for hang in range(len(H_vector)):
        tp = H_vector[hang]
        hou.append(tp)
        hou_temp = [0 for i in range((MicroEn_length*hang_size)-len(hou))]
        hou_value = hou+ hou_temp        
    feature = qian_value + center+ hou_value
    return (feature)


def get_fea_IDN_LabelN(result_filepath,filename,POS,ID,label,MicroEn_length,hang_size): 
    import os
    feature = [] 
    IDName = [] 
    Labelname = []      
    POS = POS
    ID = ID 
    label = label
    hang_size = hang_size    
    for j in range(len(filename)):
        xiabiao=ID.index(filename[j])
        Position= POS[xiabiao]        
        path = result_filepath
        files= os.listdir(path) 
        f = open(path+"/"+files[j]) 
        flines=f.readlines() 
        num=int(Position)-1
        IDname=ID[xiabiao]
        labelname=label[xiabiao]        
        IDName.append(IDname)
        Labelname.append(labelname)
        pos_value = [] 
        pos_va = flines[num].strip().split('    ')
        for p in range(len(pos_va)):
            t = float(pos_va[p])
            pos_value.append(t)            
        Q_vector, H_vector = QH_vetor(flines,num,MicroEn_length,hang_size) 
        temp_feature =  QHp_list3to1(Q_vector,pos_value,H_vector,MicroEn_length,hang_size)
        feature.append(temp_feature)
        f.close()
        
    return(feature,IDName,Labelname)


def get_fea_IDN_LabelN_PSA(result_filepath,filename,POS,ID,label,MicroEn_length,hang_size): 
    import os
    feature = [] 
    IDName = [] 
    Labelname = []  
    POS = POS
    ID = ID 
    label = label
    hang_size = hang_size     
    for j in range(len(filename)):
        xiabiao=ID.index(filename[j])
        Position= POS[xiabiao]        
        path = result_filepath
        files= os.listdir(path)
        f = open(path+"/"+files[j])
        flines=f.readlines()         
        num=int(Position)-1 
        IDname=ID[xiabiao]
        labelname=label[xiabiao]
        
        IDName.append(IDname)
        Labelname.append(labelname)
        pos_value = []  
        pos_va = flines[num].strip().split('    ')
        for p in range(len(pos_va)):
            t = float(pos_va[p])
            pos_value.append(t)            
        Q_vector, H_vector = QH_vetor(flines,num,MicroEn_length,hang_size)  
        temp_feature =  QHp_list3to1_PSA(Q_vector,pos_value,H_vector,MicroEn_length,hang_size)        
        feature.append(temp_feature)
        f.close()
        
    return(feature,IDName,Labelname)

def feature_name(string,length):
    feature_name_list = []
    for i in range(0,length):
        tp = ''
        temp =string + str(i)
        feature_name_list.append(temp)
    return (feature_name_list)


def save_file(fea_res_fpath,column_name,column_na_length, feature, IDName,Labelname ):   
    import pandas as pd
    feature_name_list = []
    feature_name_list = feature_name (column_name,column_na_length)
    fea_pd = pd.DataFrame(feature[:])
    fea_ID = pd.DataFrame(IDName[:])
    fea_Labelname = pd.DataFrame(Labelname[:])
    fea_pd.columns = [feature_name_list]
    fea_pd.insert(0,'Name',fea_ID )     
    fea_pd.insert(1,'Label',fea_Labelname)      
    fea_pd.to_csv(fea_res_fpath, index=True)

    return('文件保存成功！')


def SS_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    filename = seq_name
    POS = pos
    ID = seq_name
    label = [-1]
    file_openpath = data_path + "SS/"
    feature,IDName,Labelname = get_fea_IDN_LabelN(file_openpath,filename,POS,ID,label,29,3) 
    PSS_fea = feature
    print('---PSS_fea---',PSS_fea,len(PSS_fea))
    file_savepath = data_path + "SS/SS_59.csv"
    save_file(file_savepath,'ss',177, PSS_fea, ID,label) 

def PSSM_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    filename = seq_name
    POS = pos
    ID = seq_name
    label = [-1]
    file_openpath = data_path + "PSSM/"
    feature,IDName,Labelname = get_fea_IDN_LabelN(file_openpath,filename,POS,ID,label,29,20) 
    PSSM_fea = feature
    file_savepath = data_path + "PSSM/PSSM_59.csv"
    save_file(file_savepath,'pssm',1180, PSSM_fea, ID,label)


def PDO_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    filename = seq_name
    POS = pos
    ID = seq_name
    label = [-1]
    file_openpath = data_path + "PDO/"
    feature,IDName,Labelname = get_fea_IDN_LabelN(file_openpath,filename,POS,ID,label,29,1) 
    PDO_fea = feature
    file_savepath = data_path + "PDO/PDO_59.csv"
    save_file(file_savepath,'pdo',59, PDO_fea, ID,label) 

def PSA_extract(data_path,seq_name, seq, pos, wtAA, mutAA):
    filename = seq_name
    seq = seq 
    pos = pos 
    PSA_POS = [250] 
    ID = seq_name
    label = [-1]    
       file_openpath = data_path + "PSA/"
    feature,IDName,Labelname = get_fea_IDN_LabelN_PSA(file_openpath,filename,PSA_POS,ID,label,29,3) 
    PSA_fea = feature
    file_savepath = data_path + "PSA/PSA_59.csv"
    save_file(file_savepath,'psa',177, PSA_fea, ID,label)


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

def parse_physico_chemical_property_aaindex(fn):
    f = open(fn, 'r')
    ls = f.readlines()
    f.close()
    ls_aa = ls[-3].rstrip().split()[1:] 
    ls_p1 = ls[-2].rstrip().split() 
    ls_p2 = ls[-1].rstrip().split() 
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

def parse_substitution_matrix_aaindex(fn):    
    f=open(fn, 'r')
    ls =f.readlines()
    f.close()
    substitution_matrix = {}
    i_start=0
    for i, s in enumerate(ls):
        if s[0]!='M':continue
        i_start=i
        break
    rows = ls[i_start].rstrip().split()[3].strip(',')
    cols = ls[i_start].rstrip().split()[6].strip(',')
    if rows!=cols:
        print (rows, cols)
        raise ValueError('Matrix must be squared')
    for i in range(0,len(rows)):
        data = ls[i_start + 1 + i].rstrip().split() 
        for j in range(0,i+1):
            pair_aa = rows[i]+cols[j]
            value = float(data[j])
            substitution_matrix[pair_aa]=value
    return substitution_matrix

def parse_substitution_matrix_slim(fn):
    f = open(fn, 'r')
    ls = f.readlines()[1:] 
    f.close()
    substitution_matrix = {}
    ls_aa = ls[0].rstrip().split('\t')[1:]
    for s in ls[1:]:
        data = s.rstrip().split('\t')
        aa1 = data[0]
        for i in range(0,len(ls_aa)):
           aa2 = ls_aa[i]
           value = float(data[i+1])
           substitution_matrix[aa1+aa2]=value
    return substitution_matrix

def getAA(aa, code=1):
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

dir = "/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/resource/"
def getPhysicoChemicalProperties(dir, ext='.txt'):
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


def getBlosumScore(aa1, aa2):
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


def getPhatScore(aa1, aa2):  
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


def getSlimScore(aa1, aa2):
    global slim_matrix
    a1 = getAA(aa1)
    a2 = getAA(aa2)
    if a1+a2 in slim_matrix.keys():
        return slim_matrix[a1+a2]
    else:
        print (a1, a2)
        raise ValueError('No corresponding keys in the SLIM matrix')
    return -1

def calculateSequenceBasedDescriptor(acc, aa_wt, aa_mut, pos):
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
    return descriptor, s_log


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
    
def sequence_1D28(data_path,seq_name, seq, pos, wtAA, mutAA):
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
    descriptor=calculateSequenceBasedDescriptor(seq_name[0], wtAA[0], mutAA[0], pos)
    sequence_fea = descriptor[0]
    sequence_feaName = []
    for i in range(1,29,1):
        temp = 'fea'+str(i)
        sequence_feaName.append(temp)        
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import minmax_scale
    X = np.array(sequence_fea).reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)  
    fea_pd = pd.DataFrame(X_minmax[:],)
    fea_pd = fea_pd.T 
    fea_ID = pd.DataFrame(seq_name[:])
    label = [-1]
    fea_Labelname = pd.DataFrame(label[:])
    fea_pd.columns = [sequence_feaName] 
    fea_pd.insert(0,'Name',fea_ID )  
    fea_pd.insert(1,'Label',fea_Labelname) 
    save_pssmpath = data_path+'SeqFEA/'
    mkdir(save_pssmpath)
    file_savepath = data_path + "SeqFEA/sequencefea.csv"
    fea_pd.to_csv(file_savepath, index=True)
    print('sequencefea.csv save sucess.')    
    return 0
    

def read_test_txt(data_path,jobid):    
    seq_path = data_path + jobid + ".txt"
    seq_name = []
    seq = []
    pos = []
    wtAA = []
    mutAA = []
    with open(seq_path, "r") as f:
        lines = f.readlines()
        s_name = lines[0].strip('\n')[1:] 
        seq_name.append(s_name)        
        se = lines[1].strip('\n')
        seq.append(se)        
        po = int(lines[2].strip('\n'))
        pos.append(po)        
        wtA = lines[3].strip('\n')
        wtAA.append(wtA)        
        mutA = lines[4].strip('\n')
        mutAA.append(mutA)        
    return seq_name,seq,pos,wtAA,mutAA


def feature_extract(data_path,jobid):
    seq_name,seq,pos,wtAA,mutAA = read_test_txt(data_path,jobid)     
    SS_extract(data_path,seq_name, seq, pos, wtAA, mutAA) 
    PSSM_extract(data_path,seq_name, seq, pos, wtAA, mutAA)
    PDO_extract(data_path,seq_name, seq, pos, wtAA, mutAA)
    PSA_extract(data_path,seq_name, seq, pos, wtAA, mutAA)
    sequence_1D28(data_path,seq_name, seq, pos, wtAA, mutAA)
    import pandas as pd     
    PSSM_path = data_path + "PSSM/PSSM_59.csv"
    df_PSSM = pd.read_csv(PSSM_path) 
    SS_path = data_path + "SS/SS_59.csv"
    df_SS = pd.read_csv(SS_path) 
    df_SS = df_SS.loc[:,'ss0':'ss176']
    PDO_path = data_path + "PDO/PDO_59.csv"
    df_PDO = pd.read_csv(PDO_path) 
    df_PDO = df_PDO.loc[:,'pdo0':'pdo58']
    PSA_path = data_path + "PSA/PSA_59.csv"
    df_PSA= pd.read_csv(PSA_path)
    df_PSA = df_PSA.loc[:,'psa0':'psa176']
    SeqFEA_path = data_path + "SeqFEA/sequencefea.csv"
    df_seq =  pd.read_csv(SeqFEA_path) 
    df_seq = df_seq.loc[:,'fea1':'fea28']
    df1 = pd.concat([df_PSSM,df_SS,df_PDO,df_PSA,df_seq],axis=1,ignore_index=False)     
    file_savepath = data_path  + "PSPP1D28.csv"
    df1.to_csv(file_savepath,index=False) 
    print('PSPP1D28.csv save sucess') 
