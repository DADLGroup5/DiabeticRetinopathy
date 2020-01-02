
#libraries importing  for the project NUS
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, img_to_array  
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer    #for converting multi class label to binary labels , so highes tprob gets that class 
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import os
import cv2

import random
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD




#path to our dataset
dataset = 'G:\\Deep learning projects\\Datasets'

#path to save model
#model_path = 'C:\\Users\\Arudhra Narasimhan V\\Desktop\\deploy1'



IMAGE_DIMENSION = (96,96,3)     #3 channel rgb image 
BATCHSIZE = 32                  #hardcoded it below          
data = []                       #for parsing the image with labels , list
classes = []                    #considering our 4 classes (mild ,moderate , medium ,advanced)

imagepaths = sorted(list(paths.list_images(dataset)))       #creates sorted list of paths in terminal 
random.seed(42)
random.shuffle(imagepaths)   #shuffling 

for imgpath in imagepaths:
    try:
        image = cv2.imread(imgpath)                     #reading image at the path
        image = cv2.resize(image, (96, 96))             #resized image             
        image_array = img_to_array(image)               #convert image to array  
        data.append(image_array)                        #appended to list 
        label = imgpath.split(os.path.sep)[-2]          #well, splitting it from second last name in path ( basically on basis of folder name)
        classes.append(label)                           #adding the split name that is the label into label list 
    except Exception as e:                              
        print(e)                                        #if any image is not readable 

data = np.array(data,dtype='float')/255.0
labels = np.array(classes)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)                   #converted to binary labels


#kaggle vali info 

img_rows,img_cols= 200,200 

data=np.asarray(data)                                   #converts input to array 
classes=np.asarray(labels)

from sklearn.utils import shuffle
Data,Label= shuffle(data,classes, random_state=2)
train_data=[Data,Label]                                 #first in train is data, second is label
type(train_data)



learning_rate=0.00009                                   #hardcoded below 
#batch_size to train
batch_size = 30
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


X_train = X_train.reshape(X_train.shape[0], 96,96, 3)      #no of image samples, width , height channel
X_test = X_test.reshape(X_test.shape[0], 96,96,3)

X_train = X_train.astype('float32')                        #converted to float type 
X_test = X_test.astype('float32')

X_train /= 255                                             #reduced the size for faster convergence or computation 
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization     #basically normalizez such that mean is zero , sd is 1 
from keras.layers.convolutional import Conv2D, MaxPooling2D   # 2d kernel for images , for downsampling we use maxpool2d 
from keras.layers.core import Activation, Flatten,Dense , Dropout    #activation through layers , 
from keras import backend as k    


print(y_train.shape)





model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=X_train[0].shape))   #using inline activation function 
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D())
model.add(BatchNormalization(axis=-1))
#model.add(Dropout(0.50))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D())    #for extracting dominant features , downsampling 
model.add(Dropout(0.10))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization(axis=-1))               #mean , std for last axis 
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Flatten())                                       #flatteing into i dimensional array 
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(activation='softmax', units=2))    #densely connected full of neurons so all neurons get resutl from prevous neurons 

model.compile(loss='sparse_categorical_crossentropy', optimizer = opt, metrics=["accuracy"])
model.summary()


from keras.preprocessing.image import ImageDataGenerator

# create generators  - training data will be augmented images
validationdatagenerator = ImageDataGenerator()
traindatagenerator = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,rotation_range=15,zoom_range=0.1 )

batchsize=30
train_generator=traindatagenerator.flow(X_train, y_train, batch_size=batchsize) 
validation_generator=validationdatagenerator.flow(X_test, y_test, batch_size=batchsize)


model.fit_generator(train_generator, steps_per_epoch=800, epochs=7, 
                    validation_data=validation_generator, validation_steps=200)

model.save('C:\\Users\\Arudhra Narasimhan V\\Desktop\\deploy1\\model7epoch.h5')


#test score calculation
scoreTrain=model.evaluate(X_train,y_train,verbose=0)
print("train score: " ,scoreTrain[0])
print("train accuracy :", scoreTrain[1])
scores = model.evaluate(X_test, y_test , verbose=0)
print(' test score :'  ,scores[0])   #evaluation of loss function
print('test accuracy : ' , scores[1])
                                      #epoch 6th gave the best result 
print(model.predict_classes(X_test[5:11]))  #predicting on test samples
print(y_test[5:11])   #for cross checking 








