import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

data_train = pd.read_csv('datasets/dataset_train.csv')
data_test = pd.read_csv('datasets/dataset_test.csv')

# Preprocesar los datos
dataset_train = data_train.drop('attack', axis=1).values
labels_train = data_train['attack'].values

dataset_test = data_test.drop('attack', axis=1).values
labels_test = data_test['attack'].values

# Crear el normalizador
scaler = StandardScaler()

# Ajustar el normalizador a los datos de entrenamiento y transformarlos
dataset_train = scaler.fit_transform(dataset_train)

# Utilizar el mismo normalizador para transformar los datos de entrenamiento
dataset_train = scaler.transform(dataset_train)
dataset_test = scaler.transform(dataset_test)

# Convertir los datos a tensores
dataset_train = torch.FloatTensor(dataset_train).to('cpu')
dataset_test = torch.FloatTensor(dataset_test).to('cpu')
labels_train = torch.FloatTensor(labels_train).to('cpu')
labels_test = torch.FloatTensor(labels_test).to('cpu')

labels_train = labels_train[:, None]
labels_test = labels_test[:, None]

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(dataset_train.shape[1], 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(16, 1)  # Solo una neurona en la capa de salida para clasificación binaria
        self.sigmoid = nn.Sigmoid()  # Función de activación Sigmoid para la capa de salida

    def forward(self, input):
        output = self.fc1(input)
        output = self.relu1(output)
        output = self.dropout1(output)

        output = self.fc2(output)
        output = self.relu2(output)
        output = self.dropout2(output)

        output = self.fc3(output)
        output = self.relu3(output)
        output = self.dropout3(output)

        output = self.fc4(output)
        output = self.sigmoid(output)  # Aplicar la función de activación Sigmoid en la capa de salida
        return output

def eval(data, model_name='final_model.pth'):
    # Carga el modelo
    model = MyModel()
    model.load_state_dict(torch.load(model_name))
    model.eval()

    # Normaliza los datos
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)

    # Convierte los datos a tensores de PyTorch y asegúrate de que estén en el mismo dispositivo que el modelo
    data = torch.tensor(data).float().to('cpu')  # Cambia 'cpu' a 'cuda' si estás usando una GPU

    # Haz la predicción
    with torch.no_grad():
        predictions = model(data)
        predictions = predictions.round()
        predictions = 1 - predictions

    if predictions[0].item() == 1:
        return True
    else:
        return False
