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
data_size = 200 # 0 to use every picture.

def preprocess(image):
    resize = transforms.Resize((64, 64))
    image = resize(image)
    image = np.array(image)
    # plt.imshow(image)
    # plt.pause(0.001)
    image = image.transpose(2,1,0)
    return image

class CatsAndDogsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_source, data_size = 0):
        files = os.listdir(data_dir)
        files = [os.path.join(data_dir,x) for x in files]
        if data_size < 0 or data_size > len(files):
            assert("Data size should be between 0 to number of files in the dataset")
        if data_size == 0:
            data_size = len(files)
        self.data_size = data_size
        self.files = random.sample(files, self.data_size)
        self.label_source = label_source

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        image_address = self.files[idx]
        image = Image.open(image_address)
        image = preprocess(image)
        image = torch.Tensor(image)
        image = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(image)
        file_name = image_address[:-4].split("/")[1]
        label_idx = int(file_name[4:])
        label = self.label_source[label_idx]
        label = torch.tensor(label).long()
        return image, label

labels = pd.read_csv(train_labels_filename)
lenc = LabelEncoder()
labels_numeros = labels['label'] = lenc.fit_transform(labels['label'])

train_set = CatsAndDogsDataset(data_dir = train_images_path, label_source = labels_numeros, data_size = data_size)

batch_size = 16
scale = 0.1
test_proportion = .2

trainset_size = len(train_set)
indices = list(range(trainset_size))
split = int(np.floor(test_proportion * trainset_size))
train_indices, test_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler = train_sampler, num_workers=0)
test_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler = test_sampler, num_workers=0)

class CatsAndDogsNet(nn.Module):
    def __init__(self):
        super(CatsAndDogsNet, self).__init__()

        # Convultional Net

        # Entrada: 64x64
        # Salida: 60x60
        self.conv1 = nn.Conv2d(3, 64, 5)

        # Entrada: 60x60
        # Salida: 30x30
        self.pool = nn.MaxPool2d(2, 2)

        # Entrada: 30x30
        # Salida: 24x24
        self.conv2 = nn.Conv2d(64, 32, 5)

        # Fully connected network
        self.fc1 = nn.Linear(32 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 32 * 13 * 13)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = CatsAndDogsNet()

training = False

#Función que modela el entrenamiento de la red
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

for epoch in range(0, epochs):
    print("Training...")
    #Hacemos el train con los datos que salen del loader
    train_loss = train(model, train_loader, optimizer)
    
    print("Testing...")
    #Probamos el nuevo entrenamiento sobre los datos de test
    test_loss, accuracy = test(model, test_loader)
    
    #Guardamos en nuestras listas los datos de loss obtenidos
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)
    
    #Imprimimos los resultados parciales
    print('Epoch {:d}: loss entrenamiento= {:.4f}, loss validacion= {:.4f}, exactitud={:.4%}'.format(epoch, train_loss, test_loss, accuracy))

#for param_tensor in model.state_dict():
#    print(param_tensor, "\n", model.state_dict()[param_tensor].numpy())