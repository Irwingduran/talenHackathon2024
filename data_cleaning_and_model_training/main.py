import limpieza_de_datos
import pandas as pd
import nn_model

# Limpieza de los datasets
# limpieza_de_datos.clean_file('Data_train.csv', 'dataset_train.csv')
# limpieza_de_datos.clean_file('Data_test.csv', 'dataset_test.csv')

# Entrenamiento de la red neuronal
# nn_model.train(lr=0.001, epochs=1000, prints=100)
# nn_model.print_plots()
nn_model.test()

df = pd.read_csv('SISA Trafic/TraficoSISA.csv', header=None)

# se elimina la primer fila
df = df.drop(0)

predictions = []
# Se imprime mediante un ciclo for los datos del archivo csv
for i in range(len(df)):
    # Se almacena los datos de cada fila en un arreglo
    data = df.iloc[i]
    # se almacena los datos obtenidos de nn_model.eval(data) en un arreglo
    predictions.append(nn_model.eval(data))

# Se transforma el arreglo en un DataFrame
df = pd.DataFrame(predictions)

# Se imprime el DataFrame
print(df[0].value_counts())