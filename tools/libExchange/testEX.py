import os
from exchangelib import DELEGATE, Account, Credentials, Configuration, NTLM, Message, Mailbox, HTMLBody, FileAttachment
from exchangelib.protocol import BaseProtocol, NoVerifyHTTPAdapter
import urllib3
urllib3.disable_warnings()

#存附件
def saveAttachments(attachments):
    for attachment in attachments:
        # print(attachment.name)
        save_files_path =os.path.join('/home/test/test_ex/rec/',attachment.name)# 如果是Linux需要使用os.path.join拼接路径
        with open(save_files_path, 'wb') as f:
            f.write(attachment.content)



#此句用来消除ssl证书错误，使用自签证书需加上
BaseProtocol.HTTP_ADAPTER_CLS = NoVerifyHTTPAdapter



# 输入你的域账号如example\leo
cred = Credentials('yourName', 'yourPassword')

config = Configuration(server='serverAddress (i.e: */ews/exchange.asmx)', credentials=cred, auth_type=NTLM)

a = Account(
    primary_smtp_address='yourEmailName@*.com', config=config, autodiscover=True, access_type=DELEGATE
)

print('已连接')


# 保存未读邮件中的附件
a.root.refresh()
print(a.root.tree())


a.inbox.all()
a.inbox.refresh()

inbox = a.inbox # 获取用户账户的收件箱
unread_email_list = inbox.filter(is_read=False) #获取收件箱未读邮件列表

for email in unread_email_list.order_by('-datetime_received'):
    print(email.subject) # 打印标题
    print(email.is_read) # 打印是否已读
    print(email.sender.name) #打印发件人
    print(email.datetime_received) # 打印接收时间
    saveAttachments(email.attachments)

for item in a.inbox.children:
    print('文件夹名称:'+item.name)




with open(r'/home/test/test_ex/test.txt') as f: #发送文字
    msg = f.read()


m = Message(
    account=a,
    folder=a.sent,
    subject=u'test',
    body=HTMLBody(msg),
    to_recipients=[Mailbox(email_address='sendEmailAddress')]
)


#发送附件
with open('/home/test/test_ex/test_attach.txt','rb') as f_attach:
    attach = f_attach.read()
file_attach = FileAttachment(name=u"test_attach.txt", content=attach)
m.attach(file_attach)
m.send_and_save()

print("run finish")
