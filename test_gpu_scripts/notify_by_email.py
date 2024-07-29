import smtplib
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText
import sys 

email = str(sys.argv[1])
smtp_password = str(sys.argv[2])

msg = MIMEMultipart()
msg['From'] = 'bjchen4@illinois.edu'
msg['To'] = email
msg['Subject'] = 'no-reply: zk-torch gpu testing results'

message = 'success'
with open("zktorch_gh_action.out", "r") as f:
  contents = f.readlines()

for c in contents:
  if 'error' in c:
    message = 'error found in the test'
    break

msg.attach(MIMEText(message))

mailserver = smtplib.SMTP('mail.smtp2go.com', 2525)
# identify ourselves to smtp client
mailserver.ehlo()
# secure our email with tls encryption
mailserver.starttls()
# re-identify ourselves as an encrypted connection
mailserver.ehlo()
mailserver.login('bjchen4@illinois.edu', smtp_password)

mailserver.sendmail('bjchen4@illinois.edu',email,msg.as_string())

mailserver.quit()