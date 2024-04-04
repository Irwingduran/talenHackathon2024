import pandas as pd

# Lectura del dataset de entrenamiento
df = pd.read_csv('SISA Trafic/AtaqueSISA.csv', header=None)

# Obtenemos datos de la primera fila y lo guardamos en headers
headers = df.iloc[0]

# Asignamos los headers a las columnas
df.columns = headers

# Se elimina la primera fila de datos
df = df.iloc[1:]

# Verificar si hay valores nulos en el DataFrame y eliminarlos
df = df.dropna()

# Eliminamos la columna 'No.'
df = df.drop(columns=['No.'])

# Expressión regular para obtener los segundos de la columna 'Time', ejemplo 21:29:33.023166
# y lo aplicamos a la columna 'Time'
df['Time'] = df['Time'].str.extract(r'(\d+:\d+:\d+\.\d+)')
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f').dt.second

# Convertimos la columna 'Time' a tipo float64
df['Time'] = df['Time'].astype('float64')

# Restamos los valores de tiempo siguiente con el tiempo actual de la columna 'Time'
# y lo guardamos en una nueva columna llamada 'Duration' y lo recorremos una posición hacia arriba
df['Duration'] = df['Time'].diff().shift(-1)

# Eliminanos la ultima fila
df = df.iloc[:-1]

# Movemos la columna 'Duration' a la primera posición
df = df[['Duration'] + [col for col in df.columns if col != 'Duration']]
df = df.reset_index(drop=True)

# Redondeamos la columna 'Duration' a enteros
df['Duration'] = df['Duration'].round(0).astype('int')

# Cambiamos los valores 'HTTP', 'TLSv1.2', 'TLSv1.3', 'DB-LSP-DISC', 'BROWSER', 'DHCP' por 'TCP'
df['Protocol'] = df['Protocol'].replace(['HTTP', 'TLSv1.2', 'TLSv1.3', 'DB-LSP-DISC', 'BROWSER', 'DHCP'], 'TCP')

# Cambiamos los valores 'QUIC', 'SSDP', 'MDNS', 'DNS', 'NBNS', 'LLMNR' por 'UDP'
df['Protocol'] = df['Protocol'].replace(['QUIC', 'SSDP', 'MDNS', 'DNS', 'NBNS', 'LLMNR'], 'UDP')

# Cambiamos los valores 'ICMPv6' por 'ICMP'
df['Protocol'] = df['Protocol'].replace('ICMPv6', 'ICMP')

# Cambiamos los valores 'DTLS', 'OCSP', '0x99ea' por 'OTRO'
df['Protocol'] = df['Protocol'].replace(['DTLS', 'OCSP', '0x99ea', 'ARP', 'IGMPv2', 'DB-LSP-DISC/JSON', 'RTCP',
                                         'ISAKMP', 'SNMP', 'CLDAP', 'SRVLOC', 'eDonkey', 'ASF', 'IPMB', 'Pathport',
                                         'RPC', 'Portmap', 'XDMCP', 'RIPv1', 'NFS', 'L2TP', 'RADIUS', 'L2TPv3'], 'OTRO')

# Movemos la columna 'Protocol' a la segunda posición
df = df[['Duration', 'Protocol'] + [col for col in df.columns if col != 'Duration' and col != 'Protocol']]
df = df.reset_index(drop=True)

# Cambiamos los valores 'TCP', 'UDP', 'ICMP', 'OTRO' por 0, 1, 2, 3
df['Protocol'] = df['Protocol'].replace(['TCP', 'UDP', 'ICMP', 'OTRO'], [0, 1, 2, 3])

# Creamos una columna 'Flag' con valores de 0 después de la columna 'Protocol'
df.insert(2, 'Service', 0)

# Creamos dos columnas de tipo int64 'Str_bytes' y 'Dts_bytes' con valores de 0 y las ponemos en la posición 5 y 6
df.insert(3, 'Scr_bytes', 0)
df.insert(4, 'Dts_bytes', 0)

# Convertir la columna 'Length' a tipo int64
df['Length'] = df['Length'].astype('int64')

# Obtenemos los valores de 'Destination' y 'Source' de la primera fila del DataFrame
previous_destination = df.at[0, 'Destination']
previous_source = df.at[0, 'Source']

# Establecemos 'Scr_bytes' como la columna inicial a la que se sumará la longitud
column = 'Scr_bytes'

# Sumamos la longitud de la primera fila a la columna 'Scr_bytes'
df.at[0, column] += df.at[0, 'Length']

# Iteramos sobre el DataFrame desde la segunda fila
for i in range(1, len(df)):
    # Obtenemos los valores actuales de 'Destination' y 'Source'
    current_destination = df.at[i, 'Destination']
    current_source = df.at[i, 'Source']
    # Obtenemos la longitud actual
    length = df.at[i, 'Length']
    # Si tanto 'current_destination' como 'current_source' son diferentes a sus respectivos valores anteriores
    if current_destination != previous_destination and current_source != previous_source:
        # Intercalamos los valores entre 'Scr_bytes' y 'Dts_bytes'
        if column == 'Scr_bytes':
            column = 'Dts_bytes'
        else:
            column = 'Scr_bytes'
        # Si 'current_source' es diferente a 'previous_source' y 'current_source' es diferente a 'previous_destination'
        if current_source != previous_source and current_source != previous_destination:
            # Asignamos 'Scr_bytes' a la columna
            column = 'Scr_bytes'
    # Sumamos la longitud a la columna correspondiente
    df.at[i, column] += length
    # Actualizamos los valores de 'previous_destination' y 'previous_source' para la siguiente iteración
    previous_destination = current_destination
    previous_source = current_source

# Eliminamos la columna 'Length'
df = df.drop(columns=['Length'])

# Redondeamos la columna 'Time' con dos decimales
df['Time'] = df['Time'].round(2)

# Creamos una nueva columna en la posición 5 llamada 'Count'
df.insert(5, 'Count', 0)

# Creamos una lista para almacenar los índices de las filas a eliminar
rows_to_drop = []

# Iteramos sobre el DataFrame desde la segunda fila
for i in range(1, len(df)):
    # Obtenemos los valores de 'Source' y 'Destination' de la fila actual
    current_source = df.at[i, 'Source']
    current_destination = df.at[i, 'Destination']

    # Si 'current_source' es igual a 'previous_source' y 'current_destination' es igual a 'previous_destination'
    if current_source == previous_source and current_destination == previous_destination:
        # Sumamos la cantidad de veces consecutivas que se repite el valor en las columnas 'Source' y 'Destination'
        # y el total de veces lo colocamos en la columna 'Count', si no se repite se coloca 1
        df.at[i, 'Count'] = df.at[i - 1, 'Count'] + 1

        # Añadimos el índice de la fila anterior a la lista de filas a eliminar
        rows_to_drop.append(i - 1)
    # Verificamos si la fuente actual es igual al destino anterior y el destino actual es igual a la fuente anterior
    elif current_source == previous_destination and current_destination == previous_source:
        # Si se cumple la condición, incrementamos el contador de la fila actual en uno
        df.at[i, 'Count'] = df.at[i - 1, 'Count'] + 1

        # Sumamos los bytes de la fuente de la fila anterior a los bytes de la fuente de la fila actual
        df.at[i, 'Scr_bytes'] += df.at[i - 1, 'Scr_bytes']

        # Sumamos los bytes del destino de la fila anterior a los bytes del destino de la fila actual
        df.at[i, 'Dts_bytes'] += df.at[i - 1, 'Dts_bytes']

        # Añadimos el índice de la fila anterior a la lista de filas a eliminar
        rows_to_drop.append(i - 1)
    else:
        df.at[i, 'Count'] = 1

    # Actualizamos los valores de 'previous_source' y 'previous_destination' para la siguiente iteración
    previous_source = current_source
    previous_destination = current_destination

# Eliminamos del DataFrame todas las filas con índices almacenados en la lista 'rows_to_drop'
df = df.drop(rows_to_drop)

# Duplicamos la columna 5 en dos nuevas columnas 'Srv_count' y 'Dst_host_srv_count' en las posiciones 6 y 7
df.insert(6, 'Srv_count', df['Count'])
df.insert(7, 'Dst_host_srv_count', df['Count'])

# Eliminas las columnas 'Time', 'Source', 'Destination', 'src_port', 'des_port' e 'Info'
df = df.drop(columns=['Time', 'Source', 'Destination', 'src_port', 'des_port', 'Info'])

# En la columna 'Service' cambiamos los valores 0 por 'http'
df['Service'] = df['Service'].replace(0, 'http')

# Cambiamos los valores 'http' por 4
df['Service'] = df['Service'].replace('http', 4)

# Cambiamos el tipo de dato de todas las columnas a float64
df = df.astype('float64')

df.to_csv('SISA Trafic/TraficoSISA.csv', index=False, header=True)