# coding: utf-8
    

import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.optimizers import SGD,Adam
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D,Concatenate
from keras.layers import ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation,Input
from keras.layers import Add,GlobalMaxPooling2D
from keras.layers import  LSTM,ConvLSTM2D,concatenate,GlobalMaxPooling2D
from  keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.layers.core import Lambda


import tensorflow as tf
from tensorflow import keras
def identity_block(X, f, channels):
    F1, F2, F3 = channels
    X_shortcut = X
    # main path
    # 1
    X = keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1,1), padding ='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    # 2
    X = keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    # 3 
    X = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    # skip connection
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)
    return X

def convolutional_block(X, f, channels, s=2):
    F1, F2, F3 = channels
    X_shortcut = X
    # main path
    # 1 
    X = keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    # 2
    X = keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    # 3
    X = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    #  skip connection
    X_shortcut = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), 
                                     strides=(s, s), padding='valid')(X_shortcut)
    X_shortcut = keras.layers.BatchNormalization(axis=3)(X_shortcut)
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)
    return X

PRE_PSPP_59 = open('./result/MSRes-MutP-PRE-PSPP.txt','w')
import pandas as pd
data = pd.read_excel('PRE_PSPP_59.xlsx')
y = data['Label']
X = data.loc[:,'pssm0': 'psa176']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=12)
PRE_PSPP_59_train = pd.DataFrame()
X_train.insert(0,'Label',y_train)
PRE_PSPP_59_train=X_train
PRE_PSPP_59_train.to_excel('./result/PRE_PSPP_59_train.xlsx',index = False)
PRE_PSPP_59_test = pd.DataFrame()
X_test.insert(0,'Label',y_test)
PRE_PSPP_59_test=X_test
PRE_PSPP_59_test.to_excel('./result/PRE_PSPP_59_test.xlsx',index = False)
import pandas as pd
data = pd.read_excel('./result/PRE_PSPP_59_train.xlsx')
y = data['Label']
X = data.loc[:,'pssm0':'psa176']
data = pd.read_excel('./result/PRE_PSPP_59_test.xlsx')
yy = data['Label']
XX = data.loc[:,'pssm0':'psa176']
import pandas as pd
import numpy as np
train_Features = X
train_Label = y
test_Features = XX
test_Label = yy
train_len = len(train_Features)
test_len = len(test_Features)
from numpy import array
train_Features = array(train_Features).reshape(train_len,59,27,1)
train_Label = array(train_Label).reshape(train_len)
test_Features = array(test_Features).reshape(test_len,59,27,1)
test_Label = array(test_Label).reshape(test_len)

    
import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.optimizers import SGD,Adam
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D,Concatenate
from keras.layers import ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation,Input
from keras.layers import Add,GlobalMaxPooling2D
from keras.layers import  LSTM,ConvLSTM2D,concatenate,GlobalMaxPooling2D
from  keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.layers.core import Lambda
K.clear_session()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
channels = 1
nb_classes = 2
batch_size = 32
epochs = 500
filters = 32
kernel_size = 5
pooling_size = 2
img_rows = 9
img_columns = 27
color_type = 1
input1=Input(shape=(59,27,1))

conv11 = Convolution2D(128,(3, 3), strides=(1,1), activation='relu',padding='same')(input1)
BNor11 = BatchNormalization()(conv11)
MaxPool11 = MaxPooling2D(pool_size=(3,3), strides=(1, 1),border_mode='same')(BNor11)
X11 = convolutional_block(MaxPool11, 3, [128, 128, 512])
X11 = identity_block(X11, 3, [128, 128, 512])
X11 = identity_block(X11, 3, [128, 128, 512])

conv12 = Convolution2D(128,(5, 5), strides=(1,1), activation='relu',padding='same')(input1)
BNor12 = BatchNormalization()(conv12)
MaxPool12 = MaxPooling2D(pool_size=(5,5), strides=(1, 1),border_mode='same')(BNor12)
X12 = convolutional_block(MaxPool12, 3,[128, 128, 512])
X12 = identity_block(X12, 3, [128, 128, 512])
X12 = identity_block(X12, 3, [128, 128, 512])

conv13 = Convolution2D(128,(7, 7), strides=(1,1), activation='relu',padding='same')(input1)
BNor13 = BatchNormalization()(conv13)
MaxPool13 = MaxPooling2D(pool_size=(7,7), strides=(1, 1),border_mode='same')(BNor13)
X13 = convolutional_block(MaxPool13, 3, [128, 128, 512])
X13 = identity_block(X13, 3, [128, 128, 512])
X13 = identity_block(X13, 3, [128, 128, 512])

add1 =Add()([X11,X12,X13])
BN1 = BatchNormalization()(add1)

conv21 = Convolution2D(256,(3, 3), strides=(1,1), activation='relu',padding='same')(BN1 )
BNor21 = BatchNormalization()(conv21)
MaxPool21 = MaxPooling2D(pool_size=(3,3), strides=(1, 1),border_mode='same')(BNor21)
X21 = convolutional_block(MaxPool21, 3, [256, 256,1024])
X21 = identity_block(X21, 3,[256, 256,1024])
X21 = identity_block(X21, 3, [256, 256,1024])

conv22 = Convolution2D(256,(5, 5), strides=(1,1), activation='relu',padding='same')(BN1 )
BNor22 = BatchNormalization()(conv22)
MaxPool22 = MaxPooling2D(pool_size=(5,5), strides=(1, 1),border_mode='same')(BNor22)
X22 = convolutional_block(MaxPool22, 3, [256, 256,1024])
X22 = identity_block(X22, 3, [256, 256,1024])
X22 = identity_block(X22, 3, [256, 256,1024])

conv23 = Convolution2D(256,(7, 7), strides=(1,1), activation='relu',padding='same')(BN1 )
BNor23 = BatchNormalization()(conv23)
MaxPool23 = MaxPooling2D(pool_size=(7,7), strides=(1, 1),border_mode='same')(BNor23)
X23 = convolutional_block(MaxPool23, 3, [256, 256,1024])
X23 = identity_block(X23, 3, [256, 256,1024])
X23 = identity_block(X23, 3,[256, 256,1024])

add2 =Add()([X21,X22,X23])
BN2 = BatchNormalization()(add2)

conv31 = Convolution2D(512,(3, 3), strides=(1,1), activation='relu',padding='same')(BN2)
BNor31 = BatchNormalization()(conv31)
MaxPool31 = MaxPooling2D(pool_size=(3,3), strides=(1, 1),border_mode='same')(BNor31)
X31 = convolutional_block(MaxPool31, 3, [512, 512,2048], s=2)
X31 = identity_block(X31, 3, [512, 512,2048])
X31 = identity_block(X31, 3, [512, 512,2048])

conv32 = Convolution2D(512,(5, 5), strides=(1,1), activation='relu',padding='same')(BN2)
BNor32 = BatchNormalization()(conv32)
MaxPool32 = MaxPooling2D(pool_size=(5,5), strides=(1, 1),border_mode='same')(BNor32)
X32 = convolutional_block(MaxPool32, 3, [512, 512,2048], s=2)
X32 = identity_block(X32, 3, [512, 512,2048])
X32 = identity_block(X32, 3, [512, 512,2048]) 

conv33 = Convolution2D(512,(7, 7), strides=(1,1), activation='relu',padding='same')(BN2)
BNor33 = BatchNormalization()(conv33)
MaxPool33 = MaxPooling2D(pool_size=(7,7), strides=(1, 1),border_mode='same')(BNor33)
X33 = convolutional_block(MaxPool33, 3, [512, 512,2048], s=2)
X33 = identity_block(X33, 3, [512, 512,2048])
X33 = identity_block(X33, 3, [512, 512,2048])

add3 =Add()([X31,X32,X33])
BN3 = BatchNormalization()(add3)

flat = Flatten()(BN3)
den1 = Dense(2048, activation='relu')(flat)
drop1 = Dropout(0.4)(den1)
den2 = Dense(1024, activation='relu')(drop1)
drop2 = Dropout(0.4)(den2)
den3 = Dense(512, activation='relu')(drop2)
drop3 = Dropout(0.4)(den3)
den4 = Dense(256, activation='relu')(drop3)
drop4 = Dropout(0.4)(den4)
pred = Dense(nb_classes, activation='softmax')(drop4)
model=Model(input = input1,outputs = pred)
model.summary()
SGD = SGD(lr=0.001)
model.compile(loss='categorical_crossentropy',optimizer=SGD, metrics=['accuracy'])

X_train = train_Features
y_train = train_Label
X_test = test_Features
y_test = test_Label
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test= to_categorical(y_test)
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3,verbose=1, mode='auto')
hist = model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.1,callbacks=[early_stopping])
score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])