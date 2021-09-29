
# coding: utf-8

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
    
def  feature_generate_pssm_ss_pdo(data_path,jobid, seq_name, seq, pos, wtAA, mutAA):
    data_path = data_path
    jobid = jobid
    seq_name = seq_name
    seq = seq
    fa_path = data_path + jobid + "fa4pssm.fasta" 
    fawriter = open(fa_path,'w')
    for i in range(len(seq_name)): 
        fawriter.write('>'+seq_name[i]+'\n')
        fawriter.write(seq[i]+'\n')
    fawriter.close()

    import os
    import sys
    import subprocess
    os.system('rm -rf ' + '/data1/commonuser/toolbar/pssm_sequence/*') 
    os.system('cp -rf '+ fa_path+' '+  '/data1/commonuser/toolbar/pssm_sequence/')
    os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'/data1/commonuser/toolbar/pssm.jar'+' '+'/data1/commonuser/toolbar/ 1')
    seq_name = seq_name
    temppssm_path = '/data1/commonuser/toolbar/PSSM/'+jobid+'fa4pssm/logpssm/'+seq_name[-1]+'.pssm'
    if (os.path.exists(temppssm_path)): 
        save_pssmpath = data_path+'PSSM/'
        mkdir(save_pssmpath)
    else:
        print('----pssm cannot generate----')
    for i in range(len(seq_name)):
        temp_name = seq_name[i]
        pssm_path = '/data1/commonuser/toolbar/PSSM/'+jobid+'fa4pssm/logpssm/'+temp_name+'.pssm'
        os.system('cp -rf '+ pssm_path+' '+  data_path+'PSSM/') 
    import os
    import sys
    import subprocess
    fa_path = data_path + jobid + "fa4pssm.fasta" 
    os.system('rm -rf ' + '/data1/commonuser/toolbar/pss_sequence/*') 
    os.system('cp -rf '+ fa_path+' '+  '/data1/commonuser/toolbar/pss_sequence/')
    os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'/data1/commonuser/toolbar/pss.jar'+' '+'/data1/commonuser/toolbar/ 1')
    seq_name = seq_name
    temppss_path = '/data1/commonuser/toolbar/SS/'+jobid+'fa4pssm/ss/'+seq_name[-1]+'.ss'
    if (os.path.exists(temppss_path)): 
        save_psspath = data_path+'SS/'
        mkdir(save_psspath)
    else:
        print('----pss cannot generate----')
    for i in range(len(seq_name)):
        temp_name = seq_name[i]
        pss_path = '/data1/commonuser/toolbar/SS/'+jobid+'fa4pssm/ss/'+temp_name+'.ss'
        os.system('cp -rf '+ pss_path+' '+  data_path+'SS/') 

    import os
    import sys
    import subprocess
    fa_path = data_path + jobid + "fa4pssm.fasta"
    os.system('rm -rf ' + '/data1/commonuser/toolbar/pdo_sequence/*') 
    os.system('cp -rf '+ fa_path+' '+  '/data1/commonuser/toolbar/pdo_sequence/')
    os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'/data1/commonuser/toolbar/pdo.jar'+' '+'/data1/commonuser/toolbar/ 1')
    seq_name = seq_name
    temppdo_path = '/data1/commonuser/toolbar/PDO/'+jobid+'fa4pssm/pdo/'+seq_name[-1]+'.pdo'
    if (os.path.exists(temppdo_path)): 
        save_pdopath = data_path+'PDO/'
        mkdir(save_pdopath)
    else:
        print('----pdo cannot geneate----')
    for i in range(len(seq_name)):
        temp_name = seq_name[i]
        pdo_path = '/data1/commonuser/toolbar/PDO/'+jobid+'fa4pssm/pdo/'+temp_name+'.pdo'
        os.system('cp -rf '+ pdo_path+' '+  data_path+'PDO/')       
       return 0

def cut_seq(Sequence,pos,PSA_length):
    posQ=Sequence[:int(pos)] 
    posH=Sequence[int(pos)+1:]
    posQN=''
    posHN=''
    SequenceN=''
    tempQ=''
    tempH=''    
    if(len(posQ)<int(0.5*PSA_length)  and len(posH)<int(0.5*PSA_length)):
        tempQ='X'*(int(0.5*PSA_length)-len(posQ))
        posQN=tempQ+posQ
        tempH='X'*(int(0.5*PSA_length)-len(posH)+1)
        posHN=posH +tempH         
    elif(len(posQ)>int(0.5*PSA_length)and len(posH)<int(0.5*PSA_length)):
        tempQ=posQ[-int(0.5*PSA_length):]
        posQN=tempQ
        tempH='X'*(int(0.5*PSA_length)-len(posH)+1)
        posHN=posH +tempH         
    elif(len(posQ)<int(0.5*PSA_length)and len(posH)>int(0.5*PSA_length)):
        tempQ='X'*(int(0.5*PSA_length)-len(posQ))
        posQN=tempQ+posQ
        tempH=posH[:int(0.5*PSA_length)+1]
        posHN = tempH        
    else:                                                              
        tempQ=posQ[-int(0.5*PSA_length):]
        posQN=tempQ
        tempH=posH[:int(0.5*PSA_length)+1]
        posHN = tempH     
    SequenceN = posQN +posHN  
    return (SequenceN)


def  feature_generate_psa(data_path,jobid, seq_name, seq, pos, wtAA, mutAA):
    data_path = data_path
    jobid = jobid
    seq_name = seq_name
    seq = seq
    pos = pos
    fapsa_path = data_path + jobid +  "fa4psa.fasta"
    fawriter = open(fapsa_path,'w')    
    for i in range(len(seq_name)):
        PSA_length=500
        seq4psa = cut_seq(seq[i],pos[i],PSA_length)            
        fawriter.write('>'+seq_name[i]+'\n')
        fawriter.write(seq4psa+'\n')
    fawriter.close()
    print('fa4psa.fasta save sucess.')
    import os
    import sys
    import subprocess
    os.system('rm -rf ' + '/data1/commonuser/toolbar/psa_sequence/*') 
    os.system('cp -rf '+ fapsa_path+' '+  '/data1/commonuser/toolbar/psa_sequence/')
    os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'psa.jar'+' '+'./ 1')
    seq_name = seq_name
    temppsa_path = '/data1/commonuser/toolbar/PSA/'+jobid+'fa4psa/psa/'+seq_name[-1]+'.prsa'
    if (os.path.exists(temppsa_path)): 
        save_psapath = data_path+'PSA/'
        mkdir(save_psapath)
    else:
        print('----psa cannot generate----')
    for i in range(len(seq_name)):
        temp_name = seq_name[i]
        psa_path = '/data1/commonuser/toolbar/PSA/'+jobid+'fa4psa/psa/'+temp_name+'.prsa'
        os.system('cp -rf '+ psa_path+' '+  data_path+'PSA/')              
    return 0

def feature_generate(data_path,jobid, seq_name, seq, pos, wtAA, mutAA):    
    feature_generate_pssm_ss_pdo(data_path,jobid,seq_name, seq, pos, wtAA, mutAA)    
    feature_generate_psa(data_path,jobid,seq_name, seq, pos, wtAA, mutAA)    
    return 0

