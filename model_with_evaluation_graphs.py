
#libraries importing  for the project NUS
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score , roc_curve , confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import os
import cv2
import pandas as pd
import seaborn as sn

import random
from pandas_ml import ConfusionMatrix
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

#path to our dataset
dataset = 'C:/train/'

#path to save model
model_path = 'model.h5'

learning_rate = 1e-3            #hardcoded below 
HARD_EPOCHS = 100               #hardcoded below 
IMAGE_DIMENSION = (96,96,3)     #3 channel rgb image 
BATCHSIZE = 32                  #hardcoded it below          
data = []                       #for parsing the image with labels , list
classes = []                    #considering our 4 classes (mild ,moderate , medium ,advanced)

imagepaths = sorted(list(paths.list_images(dataset)))   
random.seed(42)
random.shuffle(imagepaths)
#print(imagepaths)

for imgpath in imagepaths:
    try:
        image = cv2.imread(imgpath)
        image = cv2.resize(image, (96, 96))
        image_array = img_to_array(image)    
        data.append(image_array)
        label = imgpath.split(os.path.sep)[-2]
        classes.append(label)
    except Exception as e:
        print(e)

data = np.array(data,dtype='float')/255.0
labels = np.array(classes)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)


#kaggle vali info 

img_rows,img_cols= 200,200 

data=np.asarray(data)
classes=np.asarray(labels)

from sklearn.utils import shuffle
Data,Label= shuffle(data,classes, random_state=2)
train_data=[Data,Label]
type(train_data)

learning_rate=0.00009
#batch_size to train
batch_size = 10
# number of output classes
nb_classes = 2
# number of epochs to train
nb_epoch = 100

opt= SGD(lr=learning_rate, decay= learning_rate/ nb_epoch)
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
(X, y) = (train_data[0],train_data[1])



from sklearn.model_selection import train_test_split

# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

print(X_train.shape)
print(X_test.shape)

#X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.reshape(X_train.shape[0], 96,96, 3)
X_test = X_test.reshape(X_test.shape[0], 96,96,3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten,Dense , Dropout
from keras import backend as k 


print(y_train.shape)


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=X_train[0].shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.50))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.10))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(activation='softmax', units=2))

model.compile(loss='sparse_categorical_crossentropy', optimizer = opt, metrics=["accuracy"])
model.summary()


from keras.preprocessing.image import ImageDataGenerator

# create generators  - training data will be augmented images
validationdatagenerator = ImageDataGenerator()
traindatagenerator = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,rotation_range=15,zoom_range=0.1 )

batchsize=8
train_generator=traindatagenerator.flow(X_train, y_train, batch_size=batchsize) 
validation_generator=validationdatagenerator.flow(X_test, y_test,batch_size=batchsize)

ns_probs = [0 for _ in range(len(y_test))]

history = model.fit_generator(train_generator, steps_per_epoch=2, epochs=1, 
                    validation_data=validation_generator, validation_steps=1)

#ROC Curve code
y_pred = model.predict(X_test)

#Keeping only positive outcome probabilities
y_pred = y_pred[:, 1]

ns_fpr , ns_tpr , thresholds = roc_curve(y_test , ns_probs)
fpr , tpr , thresholds = roc_curve(y_test , y_pred)

plt.plot(ns_fpr , ns_tpr , linestyle='--' , label = 'No Skill')
plt.plot(fpr , tpr , marker = '.' , label = 'Built model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('ROC.png')
plt.show()

#AUC Score
lr_auc = roc_auc_score(y_test, y_pred)
print('AUC Score : %.3f' % (lr_auc))

#Training accuracy vs Validation accuracy
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

plt.plot(training_accuracy)
plt.plot(validation_accuracy)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('AccuracyVS.png')
plt.show()

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

#Training loss vs Validation loss
plt.plot(training_loss)
plt.plot(validation_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('LossVS.png')
plt.show()
