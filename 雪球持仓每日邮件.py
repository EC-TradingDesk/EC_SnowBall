#!/usr/bin/env python
# coding: utf-8

# In[14]:


#coding = utf-8
import smtplib  # 负责发送邮件
from email.mime.text import MIMEText  # 构造文本
from email.mime.image import MIMEImage  # 构造图片
from email.mime.multipart import MIMEMultipart  # 将多个集合对象集合起来
from email.header import Header
import datetime
import warnings
warnings.filterwarnings('ignore')
# 输入发件人昵称、收件人昵称、主题，正文，附件地址,附件名称生成一封邮件
def create_email(sender_name, receiver_name, email_Subject, email_text, annex_file,annex_name):
    #生成一个空的带附件的邮件实例
    message = MIMEMultipart()
    #将正文以text的形式插入邮件中(参数1：正文内容，参数2：文本格式，参数3：编码方式)
    message.attach(MIMEText(email_text, 'plain', 'utf-8'))
    #生成发件人名称
    message['From'] = Header(sender_name, 'utf-8')
    #生成收件人名称
    message['To'] = Header(receiver_name, 'utf-8')
#     message['To'] = receiver
    #生成邮件主题
    message['Subject'] = Header(email_Subject, 'utf-8')
    #读取附件的内容
    att1 = MIMEText(open(annex_file, 'rb').read(), 'base64', 'utf-8')
    att1["Content-Type"] = 'application/octet-stream'
    #生成附件的名称
#     att1["Content-Disposition"] = 'attachment; filename=' + annex_name
    att1.add_header('Content-Disposition','attachment',filename = ('gbk','',annex_name))
    #将附件内容插入邮件中
    message.attach(att1)
    #返回邮件
    return message
def send_email(sender, password, receiver, msg, mail_host = "smtp.qq.com"):
    # 一个输入邮箱、密码、收件人、邮件内容发送邮件的函数
    try:
        #找到你的发送邮箱的服务器地址，已加密的形式发送
        server = smtplib.SMTP()  # 发件人邮箱中的SMTP服务器
        server.connect(mail_host, 587)
        server.ehlo()
        #登录你的账号
        server.login(sender, password)  # 括号中对应的是发件人邮箱账号、邮箱密码
        #发送邮件
        server.sendmail(sender, receiver, msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号（是一个列表）、邮件内容
        print("邮件发送成功" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        # 关闭SMTP对象
        server.quit()
    except Exception:
        print('邮件发送失败'+ datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == '__main__':
    # SMTP服务器,这里使用qq邮箱
    # mail_host = "smtp.qq.com"
    # 发件人邮箱
    sender = "541211968@qq.com"
    sender_name = "zackzhang<541211968@qq.com>"
    # 邮箱授权码,注意这里不是邮箱密码！！
    mail_license = "yjtvueymncazbajd"
    #收件人邮箱（前面的昵称一定要用英文的）
    receiver_name = "kangma1012@163.com;115053043@qq.com;zz2705@columbia.edu;541211968@qq.com"
    receiver = "kangma1012@163.com,115053043@qq.com,zz2705@columbia.edu,541211968@qq.com".split(',')
#     receiver_name = "hao<115053043@qq.com>,zzq<kangma1012@163.com>"  #
    # 邮件主题
    subject_content = """雪球每日持仓情况{}""".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
    # 邮件正文
    body_content = """Dear, \n\n这是今天的雪球持仓情况，请查收！\n\nKind regards, \nZack"""
    annex_file = r'雪球簿记交易对冲参数_v5.xlsx'
    annex_name = "SnowBall_Daily_Valuation_{}.xlsx".format(datetime.datetime.now().strftime('%Y-%m-%d'))
    message = create_email(sender_name=sender_name,receiver_name=receiver_name,email_Subject=subject_content, \
                 email_text=body_content,annex_file=annex_file,annex_name=annex_name)
    send_email(sender=sender,password=mail_license,receiver=receiver,msg=message)


# In[ ]:




