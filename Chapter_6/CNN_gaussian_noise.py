#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


os.getcwd()


# In[3]:


os.listdir()


# In[4]:


data_dir = '/home/ankit/Documents/Anchita_Thesis/Chapter_6_new/data' 
#os.chdir(data_dir)
classes = os.listdir(data_dir + "/asl_alphabet_train")
num_classes = len(classes)
num_classes 


# In[5]:


class AddGaussianNoise(object):
    def __init__(self, mean, std):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        
        return tensor + torch.normal(self.mean, self.std, size=[3, 200, 200])
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# In[6]:


train_transform = transforms.Compose([
    #transforms.Resize(28,28),
    transforms.RandomCrop(200, padding=25, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.1)
])


# In[7]:


#load complete training dataset
dataset = ImageFolder(data_dir+'/asl_alphabet_train')
dataset


# In[8]:


#finding the size of validation and training set
test_size = int(0.10 * len(dataset))
val_size = int(0.15 * len(dataset))
train_size = len(dataset) - val_size - test_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
len(train_dataset), len(val_dataset), len(test_dataset)


# In[9]:


train_dataset.dataset.transform = train_transform


# In[10]:


val_dataset.dataset.transform = train_transform


# In[11]:


batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
#train_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)


# In[12]:


val_dataloader = DataLoader(val_dataset, batch_size)


# In[13]:


img, label = train_dataset[100]
print(img.shape)
print('Label:', dataset.classes[label])
plt.imshow(img.permute(1,2,0))


# In[14]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[15]:


train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []


# In[16]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    
    # Early stopping
    score = 0.0
    best_score = 0
    patience = 2
    counter = 0
    delta = 0.1
    
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
            
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model
            


# In[17]:


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.bn1 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.dropout1 = nn.Dropout2d(0.2)
        
        self.conv2 = nn.Conv2d(128, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.dropout2 = nn.Dropout2d(0.2)
        
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        
        #self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(16928, 256)
        self.dropout3 = nn.Dropout2d(0.2)
        self.fc2 = nn.Linear(256, 29)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x


# In[18]:


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[20]:


# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# Decay LR by a factor of 0.1 every 7 epochs
# Learning rate scheduling should be applied after optimizer’s update
# e.g., you should write your code this way:
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()

step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.5)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=20)


# In[ ]:





# In[ ]:





# In[21]:


train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

def predict_image(img, model):
    # Convert to a batch of 1 (1, 3, 200, 200)
    w = img.unsqueeze(dim=0)
    xb = w.to(device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    #return dataset.classes[preds[0].item()]
    return dataset.classes[preds[0].item()]

test_transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset.dataset.transform = test_transform
test_dataloader = DataLoader(test_dataset, batch_size)


# In[22]:


x = len(test_dataset)
corr = 0
incorrect = 0
for i in range(x):
    img, label = test_dataset[i]
    
    if(predict_image(img, model) == dataset.classes[label]):
        corr = corr+1
    else:
        incorrect = incorrect+1


# In[ ]:


accuracy_ach = corr/(incorrect+corr)
accuracy_ach


# In[ ]:


def predict_image_1(img, model):
    # Convert to a batch of 1 (1, 3, 200, 200)
    w = img.unsqueeze(dim=0)
    xb = w.to(device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


# In[ ]:


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
plt.savefig('Ch_6_CNN_no_noise_confusion_matrix.eps', format='eps')


# In[ ]:


val_accuracy_list = []
train_accuracy_list = []
for ii in range(0, len(val_accuracy)):
    train_accuracy_list.append(np.float_(val_accuracy[ii].cpu().numpy()))
    val_accuracy_list.append(np.float_(train_accuracy[ii].cpu().numpy()))


# In[ ]:


epochs = [i for i in range(20)]

plt.figure(figsize=(8,6))
plt.plot(epochs , train_accuracy_list , 'rd-' , label = 'Training Accuracy')
plt.plot(epochs , val_accuracy_list , 'gs-' , label = 'Validation Accuracy')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig('Ch_6_CNN_no_noise_accuracy.eps', format='eps')


# In[ ]:


epochs = [i for i in range(20)]

plt.figure(figsize=(8,6))
plt.plot(epochs , val_loss , 'rd-' , label = 'Training Loss')
plt.plot(epochs , train_loss , 'gs-' , label = 'Validation Loss')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.savefig('Ch_6_CNN_no_noise_loss.eps', format='eps')


# In[ ]:




