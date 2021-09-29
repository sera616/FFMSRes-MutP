
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
    fawriter.write('>'+seq_name+'\n')
    fawriter.write(seq+'\n')
    fawriter.close()
    print('fa4pssm.fasta save sucess')   
    import os
    import sys
    import subprocess
    os.system('rm -rf ' + '/data1/commonuser/toolbar/pssm_sequence/*') 
    os.system('cp -rf '+ fa_path+' '+  '/data1/commonuser/toolbar/pssm_sequence/')
    os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'/data1/commonuser/toolbar/pssm.jar'+' '+'/data1/commonuser/toolbar/ 1')
    seq_name = seq_name
    temp_name = seq_name.strip()
    pssm_path = '/data1/commonuser/toolbar/PSSM/fa4pssm/logpssm/'+temp_name+'.pssm'
    if (os.path.exists(pssm_path)):
        save_pssmpath = data_path+'PSSM/'
        mkdir(save_pssmpath)
        os.system('cp -rf '+ pssm_path+' '+  data_path+'PSSM/') 
    else:
        print('----pssm cannot egnerate----')

    import os
    import sys
    import subprocess
    fa_path = data_path + jobid + "fa4pssm.fasta" 
    os.system('rm -rf ' + '/data1/commonuser/toolbar/pss_sequence/*') 
    os.system('cp -rf '+ fa_path+' '+  '/data1/commonuser/toolbar/pss_sequence/')
    os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'/data1/commonuser/toolbar/pss.jar'+' '+'/data1/commonuser/toolbar/ 1')
    seq_name = seq_name
    temp_name = seq_name.strip()
    pss_path = '/data1/commonuser/toolbar/SS/fa4pssm/ss/'+temp_name+'.ss'
    if (os.path.exists(pss_path)):
        save_pssmpath = data_path+'SS/'
        mkdir(save_pssmpath)
        os.system('cp -rf '+ pss_path+' '+  data_path+'SS/')
    else:
        print('----pss canot generate----')

    import os
    import sys
    import subprocess
    fa_path = data_path + jobid + "fa4pssm.fasta" 
    os.system('rm -rf ' + '/data1/commonuser/toolbar/pdo_sequence/*') 
    os.system('cp -rf '+ fa_path+' '+  '/data1/commonuser/toolbar/pdo_sequence/')
    os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'/data1/commonuser/toolbar/pdo.jar'+' '+'/data1/commonuser/toolbar/ 1')
    seq_name = seq_name
    temp_name = seq_name.strip()
    pdo_path = '/data1/commonuser/toolbar/PDO/fa4pssm/pdo/'+temp_name+'.pdo'
    if (os.path.exists(pdo_path)):
        save_pssmpath = data_path+'PDO/'
        mkdir(save_pssmpath)
        os.system('cp -rf '+ pdo_path+' '+  data_path+'PDO/')
    else:
        print('----pdo cannot generae----')    
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
    PSA_length=500
    seq4psa = cut_seq(seq,pos,PSA_length) 
    fapsa_path = data_path + jobid +  "fa4psa.fasta"
    fawriter = open(fapsa_path,'w')
    fawriter.write('>'+seq_name+'\n')
    fawriter.write(seq4psa+'\n')
    fawriter.close()
    import os
    import sys
    import subprocess
    os.system('rm -rf ' + '/data1/commonuser/toolbar/psa_sequence/*') 
    os.system('cp -rf '+ fapsa_path+' '+  '/data1/commonuser/toolbar/psa_sequence/')
    os.system('cd /data1/commonuser/toolbar/'+';'+'java'+' '+'-jar'+' '+'psa.jar'+' '+'./ 1')
    seq_name = seq_name
    temp_name = seq_name.strip()
    psa_path = '/data1/commonuser/toolbar/PSA/fa4psa/psa/'+temp_name+'.prsa'
    if (os.path.exists(psa_path)):
        save_pssmpath = data_path+'PSA/'
        mkdir(save_pssmpath)
        os.system('cp -rf '+ psa_path+' '+  data_path + 'PSA/')
    else:
        print('----psa cannot generate----')
    return 0

def feature_generate(data_path,jobid, seq_name, seq, pos, wtAA, mutAA):    
    feature_generate_pssm_ss_pdo(data_path,jobid,seq_name, seq, pos, wtAA, mutAA)    
    feature_generate_psa(data_path,jobid,seq_name, seq, pos, wtAA, mutAA)        
    return 0