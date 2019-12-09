import os
import random
import pandas as pd
import torch.utils.data.dataset
import torchvision.transforms as transforms
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from torch.utils.data.sampler import SubsetRandomSampler

# Params
train_images_path = 'train/'
train_labels_filename = 'train_labels.csv'
data_size = 500

# Transforms
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor()
])

# Dataset
class CatsAndDogsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_source, data_size = 0, transform = None):
        files = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
        if data_size < 0 or data_size > len(files):
            assert("Data size should be between 0 to number of files in the dataset")
        if data_size == 0:
            data_size = len(files)
        self.data_size = data_size
        self.files = random.sample(files, self.data_size)
        self.label_source = label_source
        self.transform = transform

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        image_address = self.files[idx]
        image = Image.open(image_address)
        image = self.transform(image)
        return image, self.__imagelabel__(image_address)

    def __imagelabel__(self, image_address):
        file_name = image_address[:-4].split("/")[1]
        label_idx = int(file_name[4:])
        label = self.label_source[label_idx]
        label = torch.tensor(label).long()
        return label

# Labels
labels = pd.read_csv(train_labels_filename)
labels_encoder = LabelEncoder()
labels_numeros = labels_encoder.fit_transform(labels['label'])

cats_and_dogs_dataset = CatsAndDogsDataset(data_dir = train_images_path, label_source = labels_numeros, data_size = data_size, transform = image_transforms)

batch_size = 84
test_proportion = .2
train_size = int((1-test_proportion) * len(cats_and_dogs_dataset))
test_size = len(cats_and_dogs_dataset) - train_size

# Datasets and Loaders
train_dataset, test_dataset = torch.utils.data.random_split(cats_and_dogs_dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# Net
class CatsAndDogsNet(nn.Module):
    def __init__(self):
        super(CatsAndDogsNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, 5)
        self.fc1 = nn.Linear(32 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 32 * 29 * 29)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x  

model = CatsAndDogsNet()

def train(model, data_loader, optimizer):
    #El modelo se debe poner en modo training
    model.train()
    train_loss = 0
    
    for batch, tensor in enumerate(data_loader):
        data, target = tensor
        #Se pasan los datos por la red y se calcula la función de loss
        optimizer.zero_grad()
        out = model(data)
        loss = loss_criteria(out, target)
        train_loss += loss.item()

        #Se hace la backpropagation y se actualizan los parámetros de la red
        loss.backward()
        optimizer.step()

    #Se devuelve el loss promedio
    avg_loss = train_loss / len(data_loader.dataset)
    return avg_loss

def test(model, data_loader):
    #Ahora ponemos el modelo en modo evaluación
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch, tensor in enumerate(data_loader):
            data, target = tensor
            #Dado el dato, obtenemos la predicción
            out = model(data)

            #Calculamos el loss
            test_loss += loss_criteria(out, target).item()

            #Calculamos la accuracy (exactitud) (Sumando el resultado como
            #correcto si la predicción acertó)
            _, predicted = torch.max(out.data, 1)
            correct += torch.sum(target==predicted).item()
            
    #Devolvemos la exactitud y loss promedio
    avg_accuracy = correct / len(data_loader.dataset)
    avg_loss = test_loss / len(data_loader.dataset)
    return avg_loss, avg_accuracy

#Definimos nuestro criterio de loss
#Aquí usamos CrossEntropyLoss, que está poensado para clasificación
loss_criteria = nn.CrossEntropyLoss()

#Definimos nuestro optimizer
#Aquí usamos Stochastic Gradient Descent (SGD) - Descenso por Gradiente Estocástico
learning_rate = 0.01
learning_momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=learning_momentum)

#En estas listas vacías nos vamos guardando el loss para los datos de training
#y validación en cada iteración.
epoch_nums = []
training_loss = []
validation_loss = []

#Entrenamiento. Por default lo hacemos por 100 iteraciones (epochs) 
epochs = 100
for epoch in range(1, epochs + 1):
    #Hacemos el train con los datos que salen del loader
    train_loss = train(model, train_loader, optimizer)
    
    #Probamos el nuevo entrenamiento sobre los datos de test
    test_loss, accuracy = test(model, test_loader)
    
    #Guardamos en nuestras listas los datos de loss obtenidos
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)
    
    #Cada 10 iteraciones vamos imprimiendo nuestros resultados parciales
    if (epoch) % 10 == 0:
        print('Epoch {:d}: loss entrenamiento= {:.4f}, loss validacion= {:.4f}, exactitud={:.4%}'.format(epoch, train_loss, test_loss, accuracy))

#Creamos la matriz de confusión, esta es parte del paquete scikit
from sklearn.metrics import confusion_matrix

#Ponemos el modelo en modo evaluación
model.eval()

#Hacemos las predicciones para los datos de test
#Para eso, en primer lugar generamos la matriz de entradas y vector de 
#resultados a partir del dataloader
entradas = list()
salidas = list()
for batch,tensor in enumerate(test_loader):   
    valor,salida = tensor
    entradas.append(valor)
    salidas.append(salida)
#Se pasan a formato Tensor
entradas = torch.cat(entradas)
salidas = torch.cat(salidas)
#Se obtienen las predicciones
_, predicted = torch.max(model(entradas), 1)

#Graficamos la matriz de confusión
cm = confusion_matrix(salidas.numpy(), predicted.numpy())
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(10)

plt.xticks(tick_marks, labels_encoder.inverse_transform(range(2)), rotation=45)
plt.yticks(tick_marks, labels_encoder.inverse_transform(range(2)))
plt.xlabel("El modelo predijo que era")
plt.ylabel("La imágen real era")
plt.show()