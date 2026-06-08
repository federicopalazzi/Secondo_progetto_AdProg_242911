"""
2026_05_08

Evaluation file: loads the model and evaluates it on the test dataset.

Convolutional Neural Network (CNN) exercise for the
Programmazione avanzata ed intelligenza artificiale [146179]
class at the University of Trento.

Objective: Application of a CNN in a toy example
"""
from pathlib import Path
from sys import exit

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from flower_dataset import FlowerDataloader, preprocess_data # dataset is defined here

print("Quale dei due modelli è stato usato per il training (alex/dense)?")
model_select = input()
if model_select=='alex':
    Alex = True
    Dense = False
elif model_select=='dense':
    Alex = False
    Dense = True
else:
    exit("Errore. Per scegliere il modello scrivere alex o dense. Terminazione dell'esecuzione")

### 1. Load Test Data
# get paths
base_path = Path(__file__).resolve().parent # project folder path
data_folder_path = base_path / "flowers"
train_folder_path = data_folder_path / "flowers_train"
test_folder_path = data_folder_path / "flowers_test"

#  load data
flowers_dataloader = FlowerDataloader(train_folder_path, test_folder_path)
(x_train, y_train), (x_test, y_test) = flowers_dataloader.load_data()

# 2. Preprocess
aug=False
test_dataset = preprocess_data(x_test, y_test, aug)
test_loader = DataLoader(test_dataset, batch_size=64, num_workers=0, shuffle=False)


# 3. Initialize Model and Load Saved Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if Alex:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
elif Dense:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    
# match the classifier used during training (10 classes)
if Alex:
    model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=10)
elif Dense:
    model.classifier = torch.nn.Linear(in_features=1024, out_features=10)

model_filepath = base_path / "alexnet_flowers.pth"
if not model_filepath.exists():
    raise FileNotFoundError(f"Saved model not found: {model_filepath}")

# load weights and move model to device
model.load_state_dict(torch.load(model_filepath, map_location=device))
model = model.to(device)
model.eval()

# 4. Evaluation Loop
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # collecting for confusion matrix
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f'Accuracy on test set ({total} images): {100 * correct / total:.2f}%')

### Extra: confusion matrix
# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
disp.plot(cmap=plt.cm.Blues, values_format='d')
if Alex:
    plt.title(f'Confusion Matrix: AlexNet on Flowers\nAccuracy: {100 * correct / total:.2f}%')
elif Dense:
    plt.title(f'Confusion Matrix: DenseNet on Flowers\nAccuracy: {100 * correct / total:.2f}%')    
plt.show()
