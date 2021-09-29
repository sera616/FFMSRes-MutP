
# coding: utf-8

# In[ ]:


#!/usr/bin/python
# -*- coding: UTF-8 -*-

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

# 第三方 SMTP 服务
mail_host="smtp.163.com"
mail_user="yulabgroup@163.com"
mail_pass="yu123456"
sender = 'yulabgroup@163.com'
receivers = []
def save_to_file(file_name, contents):
    fh = open(file_name, 'a')
    fh.write(contents)
    fh.write('\n')
    fh.close()


def sendmail(jobid, email, job_path):
    # 创建一个带附件的实例
    message = MIMEMultipart()
    message['From'] =mail_user
    message['To'] = email
    receivers.append(email)
    subject = "Predictions of FFMSRes-MutP for job id:"+ jobid +" are ready"
    message['Subject'] = Header(subject, 'utf-8')

    # 邮件正文内容
    message.attach(MIMEText('''   Dear User,
        
        Thanks for using our web. 
        Prediction results of your job (ID: '''+ jobid +''') are ready! 
        You can find the results for this job at: http://202.119.84.36:3079/ffmsresmutp/analysis/'''+ jobid +'''
        
   Pattern Recognition and Bioinformatics Group 
   Nanjing University of Science and Technology
    '''
                            , 'plain', 'utf-8'))


    # 构造附件1
    att1 = MIMEText(open(job_path + jobid + '.csv', 'rb').read(), 'base64', 'utf-8')
#     job_path = "/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/"
#     jobid = '20210422232231'
#     temp = job_path + jobid + 'att.csv'
#     print(temp) #/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/20210422232231att.csv
    att1["Content-Type"] = 'application/octet-stream'
    att1["Content-Disposition"] = 'attachment; filename="FFMSRes-MutP-predction.csv"'
    message.attach(att1)

#     # 构造附件1
#     att1 = MIMEText(open(job_path + jobid + 'att.csv', 'rb').read(), 'base64', 'utf-8')
# #     job_path = "/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/"
# #     jobid = '20210422232231'
# #     temp = job_path + jobid + 'att.csv'
# #     print(temp) #/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/20210422232231att.csv
#     att1["Content-Type"] = 'application/octet-stream'
#     att1["Content-Disposition"] = 'attachment; filename="Attention values.csv"'
#     message.attach(att1)
    
#     att2 = MIMEText(open(job_path + jobid + '.png', 'rb').read(), 'base64', 'utf-8')
#     # job_path + jobid + '.png' : '/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/20210422232231.png'
#     att2["Content-Type"] = 'application/octet-stream'
#     att2["Content-Disposition"] = 'attachment; filename="Attention image.png"'
#     message.attach(att2)

    try:
        smtpObj = smtplib.SMTP(mail_host, 25)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("send mail successfully")

    except smtplib.SMTPException:
        print("Error: can not send")
    try:
        save_to_file(job_path + '/email.txt', email)
    except Exception:
        print("error")


# In[1]:


# job_path = "/data1/commonuser/toolbar/webapps/gefang/ffms/PredictDataOFUsers/"
# jobid = '20210422232231'
# temp = job_path + jobid + 'att.csv'
# print(temp)


