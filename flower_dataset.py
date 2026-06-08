import random
import struct
import os
from array import array
from typing import Tuple
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms


class FlowerDataloader(object):
    def __init__(self, train_folder_path: str, test_folder_path:str):
        self.train_folder_path = train_folder_path
        self.test_folder_path = test_folder_path

    def read_images_labels(self, images_filepath: str) -> Tuple[list, list]:
        labels = []
        image_names = []
        if os.path.isdir(images_filepath):
            image_names = [f for f in os.listdir(images_filepath) if os.path.isfile(os.path.join(images_filepath, f))] # ricorda di sistemare
            for i in image_names:
                n = i.find('_')
                temp = ''
                count = 0
                for j in i:
                    temp += j
                    count += 1
                    if count == n:
                        break
                match temp:
                    case "bougainvillea":
                        labels.append(0)
                    case "daisies":
                        labels.append(1)
                    case "garden": #rappresenterebbero le rose da giardino (garden_roses), che per semplicità vengono ettichettate con garden e basta
                        labels.append(2)
                    case "gardenias":
                        labels.append(3)
                    case "hibiscus":
                        labels.append(4)
                    case "hydrangeas":
                        labels.append(5)
                    case "lilies":
                        labels.append(6)
                    case "orchids":
                        labels.append(7)
                    case "peonies":
                        labels.append(8)
                    case "tulip":
                        labels.append(9)
                del(temp)
                del(count)

            # Return file paths instead of loading images into memory
            image_paths = [os.path.join(images_filepath, k) for k in image_names]
            return image_paths, labels
        else:
            print('Porca di quella puttana hai messo il path sbagliato')
            return [], []


    def load_data(self) -> Tuple[Tuple[list, array], Tuple[list, array]]:
            x_train, y_train = self.read_images_labels(self.train_folder_path)
            x_test, y_test = self.read_images_labels(self.test_folder_path)
            return (x_train, y_train),(x_test, y_test)   


def show_images(images: list, title_texts: list):
    cols = 5
    rows = int(len(images)/cols) + 1
    
    # We maintain a large figure size so the images don't get squashed
    plt.figure(figsize=(15,10))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        
        # Turn off the pixel axes (optional, but looks much cleaner for presentation)
        # plt.axis('off') 

        # Convert PIL Image to numpy array to avoid matplotlib conversion issues 
        plt.imshow(image)
        
        if (title_text != ''):
            # Increase fontsize for large screen visibility
            plt.title(title_text, fontsize = 15, fontweight='bold');        
        index += 1
    
    # adjust subplot to give everything breathing room
    plt.tight_layout(pad=5.0)

#show_images(images, labels)

def show_flowers(x_train: list, x_test: list, y_train: array, y_test: array):
    images_2_show = []
    titles_2_show = []
    example = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.AutoAugment()
    ])

    for i in range(0, 10):
        r = random.randint(1, 514)
        # x_train contains file paths now; open image for display
        images_2_show.append(example(Image.open(x_train[r]).convert('RGB'))) 
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

    for i in range(0, 5):
        r = random.randint(1, 219)
        images_2_show.append(example(Image.open(x_test[r]).convert('RGB')))        
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

    show_images(images_2_show, titles_2_show)
    plt.show()

class FlowerDataset(torch.utils.data.Dataset): 
    def __init__(self, image_paths, labels, transform=None, aug=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.aug = aug

    def __len__(self):
        if self.aug:
            return len(self.image_paths)*3
        else:
            return len(self.image_paths)

    def __getitem__(self, idx):
        if self.aug:
            idx = idx % len(self.image_paths)
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
    
    
def preprocess_data(x, y, aug=False, resize=(448, 448)):
    # Apply Resize + ToTensor + Normalize per-sample to avoid loading all tensors at once
    normal = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    augment = transforms.Compose([
        transforms.Resize(resize),
        transforms.AutoAugment(), #chiave di volta
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if aug:
        return FlowerDataset(x, list(y), transform=augment, aug=True)
    else:
        return FlowerDataset(x, list(y), transform=normal)

if __name__ == "__main__":
    data_folder_path = '/home/nuke/AdProgPy/flowers'
    train_folder_path = data_folder_path + '/flowers_train'
    test_folder_path = data_folder_path + '/flowers_test'


    flowers_dataloader = FlowerDataloader(train_folder_path, test_folder_path)
    (x_train, y_train), (x_test, y_test) = flowers_dataloader.load_data()
    show_flowers(x_train, x_test, y_train, y_test)





