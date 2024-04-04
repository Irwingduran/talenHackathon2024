import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuración del servidor SMTP
smtp_server = 'smtp.office365.com'
port = 587  # Puerto de conexión TLS
sender_email = 'pentest.etico80@hotmail.com'
password = 'pentest8080'

subject = 'Anomalias en la Red!!!!!!!!'
body = 'Hola, se detecto una amaomalia en la red. \n Se tomarán medidas de seguridad. \n Saludos.'

message = MIMEMultipart()
message['From'] = sender_email
message['To'] = 'pentest.etico80@hotmail.com'
message['Subject'] = subject

message.attach(MIMEText(body, 'plain'))

def send():
    print("Enviando correo...")
    # Iniciar sesión en el servidor SMTP
    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls()
        #print(sender_email, password)
        server.login(sender_email, password)
        text = message.as_string()
        server.sendmail(sender_email, 'pentest.etico80@hotmail.com', text)
