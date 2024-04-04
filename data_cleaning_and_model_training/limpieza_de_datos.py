import pandas as pd
import numpy as np


def clean_file(file_name, new_file_name):
    # Lectura del dataset de entrenamiento
    df = pd.read_csv(f'datasets/{file_name}', header=None)

    # Se elimina la primera fila de datos
    df = df.iloc[1:]

    # Se reemplazan los headers por sus nombres
    headers = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
               'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
               'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
               'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
               'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
               'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
               'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
               'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
               'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
               'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'last_flag']
    df.columns = headers

    # Se buscan los posibles datos nulos
    print(df.isnull().sum())

    # Se listan los valores únicos de la columna 'protocol_type' y se almacena la cantidad de estos valores unicos
    print(df['protocol_type'].unique())
    print(len(df['protocol_type'].unique()))

    # Se cambian los valores de la columna 'protocol_type' por valores numéricos
    df['protocol_type'] = df['protocol_type'].replace(df['protocol_type'].unique(), range(len(df['protocol_type'].unique())))

    # Se listan los valores únicos de la columna 'service' y se almacena la cantidad de estos valores unicos
    print(df['service'].unique())
    print(len(df['service'].unique()))

    # Se cambian los valores de la columna 'service' por valores numéricos
    df['service'] = df['service'].replace(df['service'].unique(), range(len(df['service'].unique())))

    # Se listan los valores únicos de la columna 'attack'
    print(df['attack'].unique())
    print(len(df['attack'].unique()))

    # se reemplazan los valores 'neptune', 'smurf', 'apache2', 'back', 'processtable', 'pod', 'worm', 'teardrop',
    # 'land', 'udpstorm' de la columna 'attack' por  'DOS'
    attacks_to_replace = ['neptune', 'smurf', 'apache2', 'back', 'processtable', 'pod', 'worm', 'teardrop', 'land',
                          'udpstorm']
    df['attack'] = df['attack'].replace(attacks_to_replace, 'DOS')

    # se reemplazan los valores 'saint', 'satan', 'mscan', 'nmap', 'ipsweep', 'portsweep', 'spy' de la columna 'attack'
    # por 'Probe'
    attacks_to_replace = ['saint', 'satan', 'mscan', 'nmap', 'ipsweep', 'portsweep', 'spy']
    df['attack'] = df['attack'].replace(attacks_to_replace, 'probe')

    # se reemplazan los valores 'guess_passwd', 'warezmaster', 'snmpgetattack', 'httptunnel', 'snmpguess', 'multihop',
    # 'named', 'xlock', 'xsnoop', 'ftp_write', 'imap', 'phf', 'sendmail', 'mailbomb', 'warezclient'
    # de la columna 'attack' por  'R2L'
    attacks_to_replace = ['guess_passwd', 'warezmaster', 'snmpgetattack', 'httptunnel', 'snmpguess', 'multihop',
                          'named', 'xlock', 'xsnoop', 'ftp_write', 'imap', 'phf', 'sendmail', 'mailbomb', 'warezclient']
    df['attack'] = df['attack'].replace(attacks_to_replace, 'R2L')

    # se reemplazan los valores 'buffer_overflow', 'ps', 'loadmodule', 'xterm', 'rootkit', 'perl', 'sqlattack'
    # de la columna 'attack' por  'U2R'
    attacks_to_replace = ['buffer_overflow', 'ps', 'loadmodule', 'xterm', 'rootkit', 'perl', 'sqlattack']
    df['attack'] = df['attack'].replace(attacks_to_replace, 'U2R')

    # se eliminan las filas donde aparecen los valores 'U2R', 'R2L' y 'probe' en la columna 'attack'
    df = df[~df['attack'].isin(['U2R', 'R2L', 'probe'])]

    # Se vuelven a listar los valores únicos de la columna 'attack' y se almacena la cantidad de estos valores unicos
    print(df['attack'].unique())
    print(len(df['attack'].unique()))
    print(df['attack'].value_counts())

    # Se cambian los valores de la columna 'attack' y se almacena la cantidad de estos valores
    df['attack'] = df['attack'].replace(df['attack'].unique(), range(len(df['attack'].unique())))

    # Eliminar columna 'wrong_fragment', 'urgent', 'num_failed_logins', 'num_compromised', 'num_shells',
    # 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_access_files',
    # 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'loggin_in', 'last_flag'
    df = df.drop(columns=['wrong_fragment', 'urgent', 'num_failed_logins', 'num_compromised',
                          'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                          'num_access_files', 'num_outbound_cmds', 'is_host_login',
                          'is_guest_login', 'logged_in', 'last_flag', 'num_shells',
                          'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                          'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_same_srv_rate',
                          'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                          'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                          'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'hot', 'flag', 'land'])

    # Verificar tipo de datos de cada columna
    print(df.dtypes)

    # Todos tipos de datos int64 se cambian a float64,excepto de las columnas eliminadas
    for column in df.columns:
        if df[column].dtype == 'int64':
            df[column] = df[column].astype('float64')
    print(f'\n{df.dtypes}')
    df.to_csv(f'datasets/{new_file_name}', index=False, header=True)
