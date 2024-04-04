import pandas as pd
import send_mail as mail
import time
import nn_model

mirroring = pd.read_csv('SISA_trafic.csv', header=None).drop(0)

# Almacenamiento de las predicciones
predictions = []

start_time = 0
elapsed_time = 0
first_time = True

# Se imprime mediante un ciclo for los datos del archivo csv
for i in range(len(mirroring)):
    # Se almacena los datos de cada fila en un arreglo
    data = mirroring.iloc[i]

    # se almacena los datos obtenidos de nn_model.eval(data) en un arreglo
    prediction = nn_model.eval(data)
    predictions.append(prediction)

    if prediction:
        elapsed_time = time.time() - start_time
        # Se envía un correo si han pasado 15 minutos desde el ultimo o es la primera vez que se detecta una alerta
        if elapsed_time >= 900 or first_time:

            mail.send()
            start_time = time.time()
            first_time = False

# Se transforma el arreglo en un DataFrame
predictions = pd.DataFrame(predictions)

# Se imprime el DataFrame
print(predictions[0].value_counts())