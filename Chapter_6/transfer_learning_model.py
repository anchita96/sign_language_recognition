#!/usr/bin/env python
# coding: utf-8

#import the required libraries
import os
import torch
import torchvision
import string
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset
from torchvision.utils import make_grid
import torchvision.models as models
get_ipython().run_line_magic('matplotlib', 'inline')
from copy import copy
import time
import copy
import tensorflow as tf
import os
import cv2



# Load the dataset
data_dir = './data' 
classes = os.listdir(data_dir + "/asl_alphabet_train")
num_classes = len(classes)


# Create a class to add Gaussian noise for transformation of images
class AddGaussianNoise(object):
    def __init__(self, mean, std):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        
        return tensor + torch.normal(self.mean, self.std, size=[3, 200, 200])
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# Perform data auguemntation by creating a transform function for training and validation datasets
# we add the gaussian noise at variance level 0.1 for traing a generalized model
train_transform = transforms.Compose([
    #transforms.Resize(28,28),
    transforms.RandomCrop(200, padding=25, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.1)
    
])


# Load complete training dataset
dataset = ImageFolder(data_dir+'/asl_alphabet_train')


# Divide the dataset into training, validation and test dataset.
test_size = int(0.10 * len(dataset))
val_size = int(0.15 * len(dataset))
train_size = len(dataset) - val_size - test_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


# Perform the data augumentation on training and validation data
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = train_transform



# Create training and validation set dataloader
batch_size = 50
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size)


# Show sample image and label
img, label = val_dataset[0]
print(img.shape)
print('Label:', dataset.classes[label])
plt.imshow(img.permute(1,2,0))
plt.show()


# Show sample batch of images
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:49], nrow=7).permute(1, 2, 0))
        break
show_batch(train_dataloader)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_loss = [] # for saving training loss
train_accuracy = [] # for saving training acccuracy
val_loss = [] # for saving validation loss
val_accuracy = [] # for saving validation loss

# Train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    
    # Early stopping
    score = 0.0
    best_score = 0
    patience = 4
    counter = 0
    delta = 0.01
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        
        
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # backward + optimize only if in training phase
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()     

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        scheduler.step()
        train_epoch_loss = running_loss / len(train_dataset)
        train_epoch_acc = running_corrects.double() / len(train_dataset)

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_epoch_loss, train_epoch_acc))
        
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_acc)
        
        model.eval()   # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0
        
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        
        val_epoch_loss = running_loss / len(val_dataset)
        val_epoch_acc = running_corrects.double() / len(val_dataset)
        
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(val_epoch_loss, val_epoch_acc))
        
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_acc)
        
        # deep copy the model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
#         # Early stopping
#         score = val_epoch_acc

#         #for first run
#         if best_score is None:
#             best_score = score
#         #if this epoch accuracy is less than the (previous one + some value)
#         elif score < best_score + delta:
#             counter += 1
#             if counter >= patience:
#                 print('Early stopping!')
#                 return model
#         else:
#             best_score = score
#             counter = 0

#         last_accuracy = val_epoch_acc
            
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model
            


# Load a pretrained model and reset final fully connected layer.

model = models.resnet18(pretrained=True) #resnet18
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 29)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# Decay LR by a factor of 0.1 every 7 epochs

step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=20)

# Randomly split the training, validation, test size 
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Function for prediction of the testing set
def predict_image(img, model):
    # Convert to a batch of 1 (1, 3, 200, 200)
    w = img.unsqueeze(dim=0)
    xb = w.to(device)
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

# Perform data augumentation in test set
test_transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.1) 
    # change the added gaussian noise to 0 and 0.01 for testing the generalizability of model on varying noise levels
])

test_dataset.dataset.transform = test_transform
test_dataloader = DataLoader(test_dataset, batch_size)

# Count the correct and incorrect predictions in the testing set.

x = len(test_dataset)
corr = 0
incorrect = 0
for i in range(x):
    img, label = test_dataset[i]
    if(predict_image(img, model) == dataset.classes[label]):
        corr = corr+1
    else:
        incorrect = incorrect+1


accuracy = corr/(corr+incorrect)


def predict_image_1(img, model):
    # Convert to a batch of 1 (1, 3, 200, 200)
    w = img.unsqueeze(dim=0)
    xb = w.to(device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


# Create the confusion matrix
x = len(test_dataset)
actual_label = np.zeros([len(test_dataset)])
pred_label = np.zeros([len(test_dataset)])
for i in range(x):
    img, label = test_dataset[i]
    actual_label[i] = label
    pred_label[i] = predict_image_1(img, model)

from sklearn.metrics import classification_report,confusion_matrix

cf_matrix = confusion_matrix(actual_label, pred_label)
cm = pd.DataFrame(cf_matrix , index = [dataset.classes[i] for i in range(29)] , columns = [dataset.classes[i] for i in range(29)])


import seaborn as sns
plt.figure(figsize = (15,15))
s = sns.heatmap(cm, cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')
s.set_xlabel('Actual label', fontsize=12)
s.set_ylabel('Predicted label', fontsize=12)

# plt.savefig('resnet_0p1.pdf')
plt.show()

# Plot the training and validation accuracy and loss

val_accuracy_list = []
train_accuracy_list = []
for ii in range(0, len(val_accuracy)):
    train_accuracy_list.append(np.float_(train_accuracy[ii].cpu().numpy()))
    val_accuracy_list.append(np.float_(val_accuracy[ii].cpu().numpy()))
    
epochs = [i for i in range(1)]
fig , ax = plt.subplots(1,2)
fig.set_size_inches(16,9)

ax[0].plot(epochs , train_accuracy_list , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_accuracy_list , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
ax[1].set_title('Training & Validation Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()