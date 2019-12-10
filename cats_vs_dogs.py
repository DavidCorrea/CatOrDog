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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data.sampler import SubsetRandomSampler

# Params
train_images_path = 'train/'
train_labels_filename = 'train_labels.csv'
data_size = 50

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
test_proportion = .09
dataset_length = len(cats_and_dogs_dataset)
train_size = int((1 - test_proportion) * dataset_length)
test_size = dataset_length - train_size

# Datasets and Loaders
train_dataset, test_dataset = torch.utils.data.random_split(cats_and_dogs_dataset, [train_size, test_size]) # labels_train, labels_test
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
    model.train()
    train_loss = 0
    
    for _, tensor in enumerate(data_loader):
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
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for _, tensor in enumerate(data_loader):
            data, target = tensor
            #Dado el dato, obtenemos la predicción
            out = model(data)

            #Calculamos el loss
            test_loss += loss_criteria(out, target).item()

            #Calculamos la accuracy (exactitud) (Sumando el resultado como correcto si la predicción acertó)
            _, predicted = torch.max(out.data, 1)
            correct += torch.sum(target == predicted).item()
            
    #Devolvemos la exactitud y loss promedio
    avg_accuracy = correct / len(data_loader.dataset)
    avg_loss = test_loss / len(data_loader.dataset)
    return avg_loss, avg_accuracy, correct

#Definimos nuestro criterio de loss
#Aquí usamos CrossEntropyLoss, que está poensado para clasificación
loss_criteria = nn.CrossEntropyLoss()

#Se define el optimizer usando Stochastic Gradient Descent (SGD) - Descenso por Gradiente Estocástico
learning_rate = 0.01
learning_momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=learning_momentum)

#En estas listas vacías nos vamos guardando el loss para los datos de training y validación en cada iteración.
epoch_nums = []
training_loss = []
validation_loss = []
hits = []

#Entrenamiento. Por default lo hacemos por 100 iteraciones (epochs) 
epochs = 5
for epoch in range(1, epochs + 1):
    #Hacemos el train con los datos que salen del loader
    train_loss = train(model, train_loader, optimizer)
    
    #Probamos el nuevo entrenamiento sobre los datos de test
    test_loss, accuracy, corrects = test(model, test_loader)
    
    #Guardamos en nuestras listas los datos de loss obtenidos
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)
    hits.append(corrects)
    
    print('Epoch {:d} Metrics - Training Loss: {:.4f} | Validation Loss: {:.4f} | Accuracy: {:.4%}'.format(epoch, train_loss, test_loss, accuracy))

print('Final Metrics -  Accuracy: {:d} | Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f}'.format(
    accuracy_score(list(test_dataset), training_loss), 
    precision_score(list(test_dataset), training_loss), 
    recall_score(list(test_dataset), training_loss), 
    f1_score(list(test_dataset), training_loss)
))