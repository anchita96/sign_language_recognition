# coding: utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # used for data and result visualization
import seaborn as sns # used for data and result visualization

import torch # library used for implementing deep learning network using pytorch framework
import torchvision
from torchvision import transforms, datasets

from sklearn.preprocessing import LabelBinarizer # used to convert data lables to one-hot encoded vectors

import keras # library used for implementing deep learning network using keras framework
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split #libraries from scikit-learn for visualizing results like confusion matrix
from sklearn.metrics import classification_report,confusion_matrix

#Read data from CSV files
train_df = pd.read_csv("sign_mnist_train.csv")
test_df = pd.read_csv("sign_mnist_test.csv")

test = pd.read_csv("sign_mnist_test.csv")
y = test['label']
pred = train_df['label']

pred.unique() #we have 25 labels (0-25) as a one-to-one map for each alphabetic letter A-Z 
# and no cases for 9=J or 25=Z because of gesture motions.


#print all alphabets
alphabet = []
for i in range(ord('A'), ord('Z') + 1):
    alphabet.append([chr(i)])
    
alphabet


# Visualizing the distribution of smaples over all class labels
class_id_distribution = train_df['label'].value_counts()
colors=['#A71930', '#DF4601', '#AB0003', '#003278', '#FF5910', 
        '#0E3386', '#BA0021', '#E81828', '#473729', '#D31145', 
        '#0C2340', '#005A9C', '#BD3039', '#EB6E1F', '#C41E3A', 
        '#33006F', '#C6011F', '#004687', '#CE1141', '#134A8E', 
        '#27251F', '#FDB827', '#0C2340', '#FD5A1E', '#00A3E0']

plt.figure(figsize=(12,7))
ax = plt.gca()
ax.set_facecolor('none')
plt.rcParams.update({'font.size': 15})
plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':True, 'top':True})
ax.spines['bottom'].set_color('black')
ax.spines['top'].set_color('black') 
ax.spines['right'].set_color('black')
ax.spines['left'].set_color('black')
plt.xticks(np.arange(43))
plt.bar(class_id_distribution.index, class_id_distribution.values, color=colors, width = 0.85)
plt.xlabel('Classes')
plt.ylabel('# Occurences in the training set')
plt.xticks(range(len(alphabet)), alphabet)
plt.grid(b=True, which='major', color='grey', linestyle='--')
# plt.savefig('Ch_5_Fig_1.eps', format='eps')
plt.show()

# This shows that the dataset is distrubuted in a balanced way as all labels have inputs.



#we seperate the labels from the dataframe and delete it from the complete dataframe
y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']


# print the columns in dataset to validate that there is no column for labels and print the first 5 rows of data frame
print(train_df.columns)
print(train_df.head())

#label_binarizer converts the input into on-hot encoding when the values are not integer. we do it for the labels.
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)


x_train = train_df.values
x_test = test_df.values


# In[13]:


print(x_train.shape) #x_train is a dataframe with 27455 rows and 784 columns. each column having its pixel

# Normalize the data
x_train = x_train / 255
x_test = x_test / 255


# Reshaping the data from 1-D to 3-D as required through input by CNN's
x_train = x_train.reshape(-1,28,28,1) # -1 mean all objects, 28*28 is the size of image, and as the MNISt dataset doenst have channel info, we add it as 1 due to greyscale.
x_test = x_test.reshape(-1,28,28,1)



train_df1 = pd.read_csv("sign_mnist_train.csv")


# code to print @University of leeds@ using the sign gesture smaples
array1 = np.zeros((10, 784))
array2 = np.zeros((2, 784))
array3 = np.zeros((5, 784))

array1[0, :] = train_df1.loc[train_df1['label'] == 20].values[0][1:]
array1[1, :]  = train_df1.loc[train_df1['label'] == 13].values[0][1:]
array1[2, :]  = train_df1.loc[train_df1['label'] == 8].values[0][1:]
array1[3, :]  = train_df1.loc[train_df1['label'] == 21].values[0][1:]
array1[4, :]  = train_df1.loc[train_df1['label'] == 4].values[0][1:]
array1[5, :]  = train_df1.loc[train_df1['label'] == 17].values[0][1:]
array1[6, :]  = train_df1.loc[train_df1['label'] == 18].values[0][1:]
array1[7, :]  = train_df1.loc[train_df1['label'] == 8].values[0][1:]
array1[8, :]  = train_df1.loc[train_df1['label'] == 19].values[0][1:]
array1[9, :]  = train_df1.loc[train_df1['label'] == 24].values[0][1:]

title1 = ['U', 'N', 'I', 'V', 'E', 'R', 'S', 'I', 'T', 'Y']
title2 = ['O', 'F']
title3 = ['L', 'E', 'E', 'D', 'S']

array2[0, :]  = train_df1.loc[train_df1['label'] == 14].values[0][1:]
array2[1, :]  = train_df1.loc[train_df1['label'] == 5].values[0][1:]

array3[0, :]  = train_df1.loc[train_df1['label'] == 11].values[0][1:]
array3[1, :]  = train_df1.loc[train_df1['label'] == 4].values[0][1:]
array3[2, :]  = train_df1.loc[train_df1['label'] == 4].values[0][1:]
array3[3, :]  = train_df1.loc[train_df1['label'] == 3].values[0][1:]
array3[4, :]  = train_df1.loc[train_df1['label'] == 18].values[0][1:]


# plot the university of leeds finger-spelling


f, ax = plt.subplots(4, 5) 
f.set_size_inches(15, 15)
k = 0
m = 0
p = 0
for i in range(0, 4):
    for j in range(0, 5):
        if i < 2:
            ax[i, j].set_title(title1[k])
            ax[i, j].imshow(array1[k].reshape(28, 28) , cmap = "gray")
            k += 1
        elif i == 2:
            if j > 0 and j < 3:
                ax[i, j].set_title(title2[m])
                ax[i, j].imshow(array2[m].reshape(28, 28) , cmap = "gray")
                m += 1
        else:
            ax[i, j].set_title(title3[p])
            ax[i, j].imshow(array3[p].reshape(28, 28) , cmap = "gray")
            p += 1
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.1)

# plt.savefig('Ch_5_Fig_2.eps', format='eps')
plt.show()


# print the sign gesture for 'E' and 'M'


array4 = train_df1.loc[train_df1['label'] == 4].values[0][1:]
plt.title('E')
plt.imshow(array4.reshape(28, 28) , cmap = "gray")

# plt.savefig('Ch_5_Fig_5a.eps', format='eps')
plt.show()


array5  = train_df1.loc[train_df1['label'] == 12].values[0][1:]
plt.title('M')
plt.imshow(array5.reshape(28, 28) , cmap = "gray")

# plt.savefig('Ch_5_Fig_5b.eps', format='eps')
plt.show()


# With data augmentation to prevent overfitting
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)



#Creating the network</h3>
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.000001)

model = Sequential()
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Flatten())
model.add(Dense(units = 256 , activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(units = 24 , activation = 'softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()


history = model.fit(datagen.flow(x_train,y_train, batch_size = 64) ,epochs = 20 , validation_data = (x_test, y_test) , callbacks = [learning_rate_reduction])


print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")


epochs = [i for i in range(20)]
train_lr = np.log10(history.history['lr'])
plt.figure(figsize=(8,6))
plt.plot(epochs , train_lr , 'go-' , label = 'Learning rate')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("$log_{10}$ (Learning rate)")
plt.grid()
plt.xticks(epochs)
# plt.savefig('Ch_5_Fig_6.eps', format='eps')
plt.show()


# Analysis 
epochs = [i for i in range(20)]
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(8,6))
plt.plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
plt.plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid()
# plt.savefig('Ch_5_Fig_3a.eps', format='eps')
plt.show()


# In[28]:


epochs = [i for i in range(20)]
train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8,6))
plt.plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
plt.plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
# plt.savefig('Ch_5_Fig_3b.eps', format='eps')
plt.show()


# plot the confusion matrix for testing the proposed model on MNIST test data
x_predict = model.predict(x_test)

predictions = np.argmax(x_predict, axis = 1)

count = 0
for i in range(7172):
    if predictions[i] >= 9:
        predictions[i] = predictions[i] +1

z = y.unique()
z.sort()

a = np.unique(predictions)
b = a.sort()

alphabet = []
for i in range(ord('A'), ord('Z') + 1):
    alphabet.append(chr(i))

alphabet1 = alphabet[0:9] + alphabet[10:25]
cm = confusion_matrix(y,predictions,normalize=None)
cm = pd.DataFrame(cm , index = [i for i in range(25) if i != 9] , columns = [i for i in range(25) if i != 9])


plt.figure(figsize = (15,15))
s=sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = .5 , annot = True, fmt='')
s.set_xlabel('Actual label', fontsize=15)
s.set_ylabel('Predicted label', fontsize=15)
s.set_xticklabels(alphabet1)
s.set_yticklabels(alphabet1)
# plt.savefig('Ch_5_Fig_4.eps', format='eps')
plt.show()



