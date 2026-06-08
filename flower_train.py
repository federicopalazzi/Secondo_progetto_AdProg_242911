"""
2026_05_08

Train file: trains the model on the train dataset and saves it in a .pth file.

Convolutional Neural Network (CNN) exercise for the
Programmazione avanzata ed intelligenza artificiale [146179]
class at the University of Trento.

Objective: Application of a CNN in a toy example
"""
from pathlib import Path
from time import time
from sys import exit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np



from flower_dataset import FlowerDataloader, preprocess_data # dataset is defined here

print("I due modelli di nn disponibili sono AlexNet e DenseNet. Quale usare per il training (alex/dense)?")
model_select = input()
if model_select=='alex':
    Alex = True
    Dense = False
elif model_select=='dense':
    Alex = False
    Dense = True
else:
    exit("Errore. Per scegliere il modello scrivere alex o dense. Terminazione dell'esecuzione")

print("Si vuole usare la data augmentation (Y/N)?")
aug_select = input()
if aug_select=='Y':
    aug = True
elif aug_select=='N':
    aug = False
else:
    exit("Errore. Scegliere tra Y o N. Terminazione dell'esecuzione")

print("Inserire numero di epoche:")
epochs = int(input())

### step 1 - load MNIST dataset

# get paths
base_path = Path(__file__).resolve().parent # project folder path
data_folder_path = base_path / "flowers"
train_folder_path = data_folder_path / "flowers_train"
test_folder_path = data_folder_path / "flowers_test"

#  load data
flowers_dataloader = FlowerDataloader(train_folder_path, test_folder_path)
(x_train, y_train), (x_test, y_test) = flowers_dataloader.load_data()

# Prepare DataLoaders
# choose device early so we can set DataLoader pin_memory accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = preprocess_data(x_train, y_train, aug)
test_dataset = preprocess_data(x_test, y_test, aug)


pin_memory = True if device.type == 'cuda' else False

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

### step 2 - train model

# define training setup
if Alex:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
elif Dense:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)


# Replace final classifier to match our 10 classes
if Alex:
    model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
elif Dense:
    model.classifier = nn.Linear(in_features=1024, out_features=10)
# Optionally freeze feature extractor to reduce memory and speed up training 
for param in model.features.parameters():
    param.requires_grad = False

# Move model to device
model = model.to(device)

if Alex:
    if aug:
        learn_rate = 0.00007
    else:
        learn_rate = 0.0001
elif Dense:
    learn_rate = 0.0004
loss_function = nn.CrossEntropyLoss()
# Only optimize parameters that require gradients (classifier head)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learn_rate)

len_set = len(train_dataset)
print(f"Training on {device} using {len_set} images with learning rate {learn_rate}...")

# count training time:
training_start_time = time()

### training loop
loss_list = []
for epoch in range(epochs):
    epoch_start_time = time()
    model.train() # Sets the model in training mode
    epoch_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) # load the data into GPU, if needed
        
        ## Forward pass
        # predict!
        outputs = model(images)
        # compute loss value
        loss = loss_function(outputs, labels)
        
        ## Backward pass and optimize
        # (batch_size == 1)
        # Clears previous gradients
        optimizer.zero_grad()
        # Computes the gradient of the loss:
        loss.backward()
        # Adjusts the weights based on the computed gradients and the learning rate:
        optimizer.step()
        
        epoch_loss += loss.item()
    
    epoch_end_time = time()
    epoch_time = epoch_end_time - epoch_start_time
    loss_list.append(epoch_loss/len(train_loader))
    print(f"Epoch [{epoch+1}/{epochs}],\tLoss: {epoch_loss/len(train_loader):.4f} \t elapsed time: {epoch_time:.4f} seconds")

training_end_time = time()
training_time = training_end_time - training_start_time
print(f"Training completed in {training_time:.4f} seconds")

plt.plot(list(range(1, len(loss_list)+1)), loss_list, linewidth=2.0)
plt.yticks(np.arange(min(loss_list), max(loss_list), 0.1))
plt.title('Loss over time')
plt.show()

### step 3 - Save the Weights (State Dict)
model_filepath = base_path / "alexnet_flowers.pth"
torch.save(model.state_dict(), model_filepath)
print(f"Model saved to {model_filepath}")
