
# coding: utf-8

# In[ ]:


# ---curPath:  /data1/commonuser/toolbar/webapps/gefang/ffms/code
# ---rootPath:  /data1/commonuser/toolbar/webapps/gefang/ffms
# ---data_path:  /data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/
# ---model_path:  /data1/commonuser/toolbar/webapps/gefang/ffms/code/model/PRE_59_27_1-28D-FFMS-Resnet.h5
    
# ---curPath:  /data1/commonuser/toolbar/webapps/gefang/ffms/code
# ---rootPath:  /data1/commonuser/toolbar/webapps/gefang/ffms
# ---data_path:  /data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/
# ---model_path:  /data1/commonuser/toolbar/webapps/gefang/ffms/code/model/PRE_59_27_1-28D-FFMS-Resnet.h5


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
    feature_generate_pssm_ss_pdo(data_path,seq_name, seq, pos, wtAA, mutAA)：
    
        使用pssm, pss, pdo 三种工具去预测特征
    
    return ()
        
'''
    
def  feature_generate_pssm_ss_pdo(data_path,jobid, seq_name, seq, pos, wtAA, mutAA):
    data_path = data_path
    jobid = jobid
    seq_name = seq_name
    seq = seq
            
    ## 2.1 保存为fasta文件，为pssm,pss, pdo运行做准备
    fa_path = data_path + jobid + "fa4pssm.fasta" 
    ##fa_path = '/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/20210422232231/20210422232231fa4pssm.fasta'
    fawriter = open(fa_path,'w')
    fawriter.write('>'+seq_name+'\n')
    fawriter.write(seq+'\n')
    fawriter.close()
    print('fa4pssm.fasta保存成功！')
    
    ## ---------------------- 计算pssm特征 -------------------------##
    import os
    import sys
    import subprocess
    print(fa_path)
    # 将上面的文件拷贝到pssm, pss, pdo存放sequence的文件夹
    os.system('rm -rf ' + '/data1/commonuser/toolbar/pssm_sequence/*') ##先删除pssm_sequence文件夹中的文件
    print('---rm 成功---')

    # subprocess.call(["cp -rf "+ fa_path+" "+ "/data1/commonuser/toolbar/pssm_sequence/"],shell=True)
    os.system('cp -rf '+ fa_path+' '+  '/data1/commonuser/toolbar/pssm_sequence/')## 将生成的fasta文件保存到pssm_sequence/文件夹下面
    print('---cp fasta文件到pssm_sequence/下成功---')


    ## 执行pssm.jar程序
    os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'/data1/commonuser/toolbar/pssm.jar'+' '+'/data1/commonuser/toolbar/ 1')
    print('--- java -jar pssm.jar ./ 1 执行完成---')

    ## 查看pssm.jar是否执行完成
    ## 若执行成功，则将*.pssm 拷贝到指定文件夹中；  否则，给出提示
    seq_name = seq_name
    temp_name = seq_name.strip()
    pssm_path = '/data1/commonuser/toolbar/PSSM/fa4pssm/logpssm/'+temp_name+'.pssm'
    print('----pssm_path----',pssm_path)

    if (os.path.exists(pssm_path)):
        #在jobid文件夹下，新建PSSM文件夹
        save_pssmpath = data_path+'PSSM/'
        mkdir(save_pssmpath)
        
        # 复制到/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/20210422232231/PSSM/
        os.system('cp -rf '+ pssm_path+' '+  data_path+'PSSM/') 
    else:
        print('----pssm特征未得到----')



    ## ---------------------- 计算pss特征 -------------------------##
    import os
    import sys
    import subprocess
    fa_path = data_path + jobid + "fa4pssm.fasta" 
    print(fa_path)
    # 将上面的文件拷贝到pssm, pss, pdo存放sequence的文件夹
    os.system('rm -rf ' + '/data1/commonuser/toolbar/pss_sequence/*') ##先删除pss_sequence文件夹中的文件
    print('---rm 成功---')

    # subprocess.call(["cp -rf "+ fa_path+" "+ "/data1/commonuser/toolbar/pssm_sequence/"],shell=True)
    os.system('cp -rf '+ fa_path+' '+  '/data1/commonuser/toolbar/pss_sequence/')## 将生成的fasta文件保存到pss_sequence/文件夹下面
    print('---cp fasta文件到pss_sequence/下成功---')


    ## 执行pss.jar程序
    os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'/data1/commonuser/toolbar/pss.jar'+' '+'/data1/commonuser/toolbar/ 1')
    print('--- java -jar pss.jar ./ 1 执行完成---')

    ## 查看pss.jar是否执行完成
    ## 若执行成功，则将*.ss 拷贝到指定文件夹中；  否则，给出提示
    seq_name = seq_name
    temp_name = seq_name.strip()
    pss_path = '/data1/commonuser/toolbar/SS/fa4pssm/ss/'+temp_name+'.ss'
    print('----pss_path----',pss_path)

    if (os.path.exists(pss_path)):
        #在jobid文件夹下，新建PSSM文件夹
        save_pssmpath = data_path+'SS/'
        mkdir(save_pssmpath)
        
        # 复制到/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/20210422232231/SS/
        os.system('cp -rf '+ pss_path+' '+  data_path+'SS/')
    else:
        print('----pss特征未得到----')


    ## ---------------------- 计算pdo特征 -------------------------##
    import os
    import sys
    import subprocess
    fa_path = data_path + jobid + "fa4pssm.fasta" 
    print(fa_path)
    # 将上面的文件拷贝到pssm, pss, pdo存放sequence的文件夹
    os.system('rm -rf ' + '/data1/commonuser/toolbar/pdo_sequence/*') ##先删除pdo_sequence文件夹中的文件
    print('---rm 成功---')

    os.system('cp -rf '+ fa_path+' '+  '/data1/commonuser/toolbar/pdo_sequence/')## 将生成的fasta文件保存到pdo_sequence/文件夹下面
    print('---cp fasta文件到pdo_sequence/下成功---')


    ## 执行pdo.jar程序
    os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'/data1/commonuser/toolbar/pdo.jar'+' '+'/data1/commonuser/toolbar/ 1')
    print('--- java -jar pdo.jar ./ 1 执行完成---')

    ## 查看pdo.jar是否执行完成
    ## 若执行成功，则将*.pdo 拷贝到指定文件夹中；  否则，给出提示
    seq_name = seq_name
    temp_name = seq_name.strip()
    pdo_path = '/data1/commonuser/toolbar/PDO/fa4pssm/pdo/'+temp_name+'.pdo'
    print('----pdo_path----',pdo_path)

    if (os.path.exists(pdo_path)):
        #在jobid文件夹下，新建PSSM文件夹
        save_pssmpath = data_path+'PDO/'
        mkdir(save_pssmpath)
        
        # 复制到/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/20210422232231/PDO/
        os.system('cp -rf '+ pdo_path+' '+  data_path+'PDO/')
    else:
        print('----pdo特征未得到----')

        
    print('----PSSM + PSS + PDO 特征都运行完毕，且已复制到目标文件夹下----')
    
    return 0


# In[ ]:


## 2.2 保存为fasta文件，为psa运行做准备

"""
cut_seq(Sequence,pos,PSA_length):
Sequence:待截取的序列
pos：突变点的位置
PSA_length:最终需要的PSA长度，目前为了便于运算，设置为?AA

"""

##----以pos为中心，截取特定长度的AA----##
def cut_seq(Sequence,pos,PSA_length):
    posQ=Sequence[:int(pos)] ## 将进来的序列，以突变点为中心，分为前后两部分
    posH=Sequence[int(pos)+1:]
    
    ##-------字符型变量使用前要先定义，后使用-----##
    posQN=''
    posHN=''
    SequenceN=''
    tempQ=''
    tempH=''
    
    if(len(posQ)<int(0.5*PSA_length)  and len(posH)<int(0.5*PSA_length)):## pos前后都小于二分之一的PSA要求长度
        tempQ='X'*(int(0.5*PSA_length)-len(posQ))
        posQN=tempQ+posQ
        tempH='X'*(int(0.5*PSA_length)-len(posH)+1)
        posHN=posH +tempH 
        
    elif(len(posQ)>int(0.5*PSA_length)and len(posH)<int(0.5*PSA_length)):## pos前大于，后小于二分之一的PSA要求长度
        tempQ=posQ[-int(0.5*PSA_length):]
        posQN=tempQ
        tempH='X'*(int(0.5*PSA_length)-len(posH)+1)
        posHN=posH +tempH 
        
    elif(len(posQ)<int(0.5*PSA_length)and len(posH)>int(0.5*PSA_length)):## pos后大于，前小于二分之一的PSA要求长度
        tempQ='X'*(int(0.5*PSA_length)-len(posQ))
        posQN=tempQ+posQ
        tempH=posH[:int(0.5*PSA_length)+1]
        posHN = tempH 
        
    else:                                                               ## pos前后都等于二分之一的PSA要求长度
        tempQ=posQ[-int(0.5*PSA_length):]
        posQN=tempQ
        tempH=posH[:int(0.5*PSA_length)+1]
        posHN = tempH        
     
    SequenceN = posQN +posHN  ##  新的序列，用于返回
    return (SequenceN)


'''
    feature_generate_psa(data_path,seq_name, seq, pos, wtAA, mutAA)：
    
        使用psa工具去预测特征
    
    return 0
       
'''
def  feature_generate_psa(data_path,jobid, seq_name, seq, pos, wtAA, mutAA):
    data_path = data_path
    jobid = jobid
    seq_name = seq_name
    seq = seq
    pos = pos

    ## 2.2 保存为fasta文件，为psa运行做准备
    PSA_length=500
    seq4psa = cut_seq(seq,pos,PSA_length) ##先调用上面的函数运行cut sequence

    fapsa_path = data_path + jobid +  "fa4psa.fasta"
    fawriter = open(fapsa_path,'w')
    fawriter.write('>'+seq_name+'\n')
    fawriter.write(seq4psa+'\n')
    fawriter.close()
    print('fa4psa.fasta保存成功！')


    ## ---------------------- 计算psa特征 -------------------------##
    import os
    import sys
    import subprocess
    print (fapsa_path)
    # 将上面的文件拷贝到psa存放sequence的文件夹
    os.system('rm -rf ' + '/data1/commonuser/toolbar/psa_sequence/*') ##先删除psa_sequence文件夹中的文件
    print('---rm 成功---')

    os.system('cp -rf '+ fapsa_path+' '+  '/data1/commonuser/toolbar/psa_sequence/')## 将生成的fasta文件保存到psa_sequence/文件夹下面
    print('---cp fasta文件到psa_sequence/下成功---')


    ## 执行psa.jar程序
    os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'psa.jar'+' '+'./ 1')
    print('--- java -jar psa.jar ./ 1 执行完成---')

    ## 查看psa.jar是否执行完成
    ## 若执行成功，则将*.psa 拷贝到指定文件夹中；  否则，给出提示
    seq_name = seq_name
    temp_name = seq_name.strip()
    psa_path = '/data1/commonuser/toolbar/PSA/fa4psa/psa/'+temp_name+'.prsa'
    print('----psa_path----',psa_path)

    if (os.path.exists(psa_path)):
        #在jobid文件夹下，新建PSSM文件夹
        save_pssmpath = data_path+'PSA/'
        mkdir(save_pssmpath)
        
        # 复制到/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/20210422232231/PSA/
        os.system('cp -rf '+ psa_path+' '+  data_path + 'PSA/')
    else:
        print('----psa特征未得到----')
        
            
    print('----PSA 特征都运行完毕，且已复制到目标文件夹下----')
    
    return 0


# In[ ]:


'''
feature_generate(data_path,jobid,seq_name, seq, pos, wtAA, mutAA):
    调用:feature_generate_pssm_ss_pdo(data_path,jobid,seq_name, seq, pos, wtAA, mutAA)
    调用: feature_generate_psa(data_path,jobid,seq_name, seq, pos, wtAA, mutAA)
    
    return 0

'''


def feature_generate(data_path,jobid, seq_name, seq, pos, wtAA, mutAA):
    
    feature_generate_pssm_ss_pdo(data_path,jobid,seq_name, seq, pos, wtAA, mutAA)
    
    feature_generate_psa(data_path,jobid,seq_name, seq, pos, wtAA, mutAA)
    
    print('---- 特征预测完成 ！----')
    
    return 0

# In[ ]:


# ## 2.1 保存为fasta文件，为pssm,pss, pdo运行做准备
# fa_path = data_path + "fa4pssm.fasta"
# fawriter = open(fa_path,'w')
# fawriter.write('>'+seq_name+'\n')
# fawriter.write(seq+'\n')
# fawriter.close()
# print('fa4pssm.fasta保存成功！')



# ## ---------------------- 计算pssm特征 -------------------------##
# import sys
# import subprocess
# print fa_path
# # 将上面的文件拷贝到pssm, pss, pdo存放sequence的文件夹
# os.system('rm -rf ' + '/data1/commonuser/toolbar/pssm_sequence/*') ##先删除pssm_sequence文件夹中的文件
# print('---rm 成功---')

# # subprocess.call(["cp -rf "+ fa_path+" "+ "/data1/commonuser/toolbar/pssm_sequence/"],shell=True)
# os.system('cp -rf '+ fa_path+' '+  '/data1/commonuser/toolbar/pssm_sequence/')## 将生成的fasta文件保存到pssm_sequence/文件夹下面
# print('---cp fasta文件到pssm_sequence/下成功---')


# ## 执行pssm.jar程序
# os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'/data1/commonuser/toolbar/pssm.jar'+' '+'/data1/commonuser/toolbar/ 1')
# print('--- java -jar pssm.jar ./ 1 执行完成---')

# ## 查看pssm.jar是否执行完成
# ## 若执行成功，则将*.pssm 拷贝到指定文件夹中；  否则，给出提示
# seq_name = seq_name
# temp_name = seq_name.strip()
# pssm_path = '/data1/commonuser/toolbar/PSSM/fa4pssm/logpssm/'+temp_name+'.pssm'
# print('----pssm_path----',pssm_path)

# if (os.path.exists(pssm_path)):
#     os.system('cp -rf '+ pssm_path+' '+  data_path)
# else:
#     print('----pssm特征未得到----')



# ## ---------------------- 计算pss特征 -------------------------##
# import sys
# import subprocess
# print fa_path
# # 将上面的文件拷贝到pssm, pss, pdo存放sequence的文件夹
# os.system('rm -rf ' + '/data1/commonuser/toolbar/pss_sequence/*') ##先删除pss_sequence文件夹中的文件
# print('---rm 成功---')

# # subprocess.call(["cp -rf "+ fa_path+" "+ "/data1/commonuser/toolbar/pssm_sequence/"],shell=True)
# os.system('cp -rf '+ fa_path+' '+  '/data1/commonuser/toolbar/pss_sequence/')## 将生成的fasta文件保存到pss_sequence/文件夹下面
# print('---cp fasta文件到pss_sequence/下成功---')


# ## 执行pss.jar程序
# os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'/data1/commonuser/toolbar/pss.jar'+' '+'/data1/commonuser/toolbar/ 1')
# print('--- java -jar pss.jar ./ 1 执行完成---')

# ## 查看pss.jar是否执行完成
# ## 若执行成功，则将*.ss 拷贝到指定文件夹中；  否则，给出提示
# seq_name = seq_name
# temp_name = seq_name.strip()
# pss_path = '/data1/commonuser/toolbar/SS/fa4pssm/ss/'+temp_name+'.ss'
# print('----pss_path----',pss_path)

# if (os.path.exists(pss_path)):
#     os.system('cp -rf '+ pss_path+' '+  data_path)
# else:
#     print('----pss特征未得到----')



# ## ---------------------- 计算pdo特征 -------------------------##
# import sys
# import subprocess
# print fa_path
# # 将上面的文件拷贝到pssm, pss, pdo存放sequence的文件夹
# os.system('rm -rf ' + '/data1/commonuser/toolbar/pdo_sequence/*') ##先删除pdo_sequence文件夹中的文件
# print('---rm 成功---')

# os.system('cp -rf '+ fa_path+' '+  '/data1/commonuser/toolbar/pdo_sequence/')## 将生成的fasta文件保存到pdo_sequence/文件夹下面
# print('---cp fasta文件到pdo_sequence/下成功---')


# ## 执行pdo.jar程序
# os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'/data1/commonuser/toolbar/pdo.jar'+' '+'/data1/commonuser/toolbar/ 1')
# print('--- java -jar pdo.jar ./ 1 执行完成---')

# ## 查看pdo.jar是否执行完成
# ## 若执行成功，则将*.pdo 拷贝到指定文件夹中；  否则，给出提示
# seq_name = seq_name
# temp_name = seq_name.strip()
# pdo_path = '/data1/commonuser/toolbar/PDO/fa4pssm/pdo/'+temp_name+'.pdo'
# print('----pdo_path----',pdo_path)

# if (os.path.exists(pdo_path)):
#     os.system('cp -rf '+ pdo_path+' '+  data_path)
# else:
#     print('----pdo特征未得到----')



# In[17]:


## 2.2 保存为fasta文件，为psa运行做准备

"""
cut_seq(Sequence,pos,PSA_length):
Sequence:待截取的序列
pos：突变点的位置
PSA_length:最终需要的PSA长度，目前为了便于运算，设置为?AA

"""

##----以pos为中心，截取特定长度的AA----##
# def cut_seq(Sequence,pos,PSA_length):
#     posQ=Sequence[:int(pos)] ## 将进来的序列，以突变点为中心，分为前后两部分
#     posH=Sequence[int(pos)+1:]
    
#     ##-------字符型变量使用前要先定义，后使用-----##
#     posQN=''
#     posHN=''
#     SequenceN=''
#     tempQ=''
#     tempH=''
    
#     if(len(posQ)<int(0.5*PSA_length)  and len(posH)<int(0.5*PSA_length)):## pos前后都小于二分之一的PSA要求长度
#         tempQ='X'*(int(0.5*PSA_length)-len(posQ))
#         posQN=tempQ+posQ
#         tempH='X'*(int(0.5*PSA_length)-len(posH)+1)
#         posHN=posH +tempH 
        
#     elif(len(posQ)>int(0.5*PSA_length)and len(posH)<int(0.5*PSA_length)):## pos前大于，后小于二分之一的PSA要求长度
#         tempQ=posQ[-int(0.5*PSA_length):]
#         posQN=tempQ
#         tempH='X'*(int(0.5*PSA_length)-len(posH)+1)
#         posHN=posH +tempH 
        
#     elif(len(posQ)<int(0.5*PSA_length)and len(posH)>int(0.5*PSA_length)):## pos后大于，前小于二分之一的PSA要求长度
#         tempQ='X'*(int(0.5*PSA_length)-len(posQ))
#         posQN=tempQ+posQ
#         tempH=posH[:int(0.5*PSA_length)+1]
#         posHN = tempH 
        
#     else:                                                               ## pos前后都等于二分之一的PSA要求长度
#         tempQ=posQ[-int(0.5*PSA_length):]
#         posQN=tempQ
#         tempH=posH[:int(0.5*PSA_length)+1]
#         posHN = tempH        
     
#     SequenceN = posQN +posHN  ##  新的序列，用于返回
#     return (SequenceN)


# ## 2.2 保存为fasta文件，为psa运行做准备
# ## 先判断seq长度是否大于9000，若小于，则直接使用pssm相同的fasta文件
# PSA_length=500
# seq4psa = cut_seq(seq,pos,PSA_length)

# fapsa_path = data_path + "fa4psa.fasta"
# fawriter = open(fapsa_path,'w')
# fawriter.write('>'+seq_name+'\n')
# fawriter.write(seq4psa+'\n')
# fawriter.close()
# print('fa4psa.fasta保存成功！')


# ## ---------------------- 计算psa特征 -------------------------##
# import sys
# import subprocess
# print fapsa_path
# # 将上面的文件拷贝到psa存放sequence的文件夹
# os.system('rm -rf ' + '/data1/commonuser/toolbar/psa_sequence/*') ##先删除psa_sequence文件夹中的文件
# print('---rm 成功---')

# os.system('cp -rf '+ fapsa_path+' '+  '/data1/commonuser/toolbar/psa_sequence/')## 将生成的fasta文件保存到psa_sequence/文件夹下面
# print('---cp fasta文件到psa_sequence/下成功---')


# ## 执行psa.jar程序
# os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'psa.jar'+' '+'./ 1')
# print('--- java -jar psa.jar ./ 1 执行完成---')

# ## 查看psa.jar是否执行完成
# ## 若执行成功，则将*.psa 拷贝到指定文件夹中；  否则，给出提示
# seq_name = seq_name
# temp_name = seq_name.strip()
# psa_path = '/data1/commonuser/toolbar/PSA/fa4psa/psa/'+temp_name+'.prsa'
# print('----psa_path----',psa_path)

# if (os.path.exists(psa_path)):
#     os.system('cp -rf '+ psa_path+' '+  data_path)
# else:
#     print('----psa特征未得到----')

