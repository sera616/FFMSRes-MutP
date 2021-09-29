
# coding: utf-8

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

mail_host="smtp.163.com"
mail_user="****p@163.com"
mail_pass="******"
sender = '****@163.com'
receivers = []
def save_to_file(file_name, contents):
    fh = open(file_name, 'a')
    fh.write(contents)
    fh.write('\n')
    fh.close()

def sendmail(jobid, email, job_path):
    message = MIMEMultipart()
    message['From'] =mail_user
    message['To'] = email
    receivers.append(email)
    subject = "Predictions of FFMSRes-MutP for job id:"+ jobid +" are ready"
    message['Subject'] = Header(subject, 'utf-8')
    message.attach(MIMEText('''   Dear User,        
        Thanks for using our web. 
        Prediction results of your job (ID: '''+ jobid +''') are ready! 
        You can find the results for this job at: http://202.119.84.36:3079/ffmsresmutp/prediction/'''+ jobid +'''        
   Pattern Recognition and Bioinformatics Group 
   Nanjing University of Science and Technology    '''
                            , 'plain', 'utf-8'))

    att1 = MIMEText(open(job_path + jobid + '.csv', 'rb').read(), 'base64', 'utf-8')
    att1["Content-Type"] = 'application/octet-stream'
    att1["Content-Disposition"] = 'attachment; filename="FFMSRes-MutP-predction.csv"'
    message.attach(att1)
    try:
        smtpObj = smtplib.SMTP(mail_host, 25)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("send mail successfully")
    except smtplib.SMTPException:
        print("Error: can not send")    try:
        save_to_file(job_path + '/email.txt', email)
    except Exception:
        print("error")

