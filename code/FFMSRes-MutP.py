
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
    # skip connection
    X_shortcut = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), 
                                     strides=(s, s), padding='valid')(X_shortcut)
    X_shortcut = keras.layers.BatchNormalization(axis=3)(X_shortcut)
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)
    return X


import tensorflow as tf
from tensorflow import keras
def identity_block_1D(X, f, channels):
    filters1, filters2, filters3 = channels
    X_shortcut = X
    # main path
    # 1
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Conv1D(filters1, 1)(X)    
    # 2
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Conv1D(filters2, kernel_size = f,padding='same')(X)
    # 3 
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Conv1D(filters3, 1)(X)
    # skip connection
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)
    return X

def convolutional_block_1D(X, f, channels, s=2):
    filters1, filters2, filters3 = channels
    X_shortcut = X
    # main path
    # 1 
    
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Conv1D(filters1, 1, strides=s)(X)
    # 2
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.Conv1D(filters2, kernel_size=f, padding='same')(X)    
    # 3
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Conv1D(filters3, 1)(X)    
    # skip connection
    X_shortcut = keras.layers.BatchNormalization()(X_shortcut)
    X_shortcut = keras.layers.Conv1D(filters3, 1, strides=s)(X_shortcut)
    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)
    return X
    
# train data loading
import pandas as pd
import numpy
from numpy import array
data = pd.read_excel('./result/PRE_28fea_59_standardfea_train.xlsx')
y_train = data['Label']
X_train_seq = data.loc[:,'pssm0':'pssm1179']
X_train_seq_Features = X_train_seq
X_train_seq_len = len(X_train_seq_Features)
X_train_seq_Features = array(X_train_seq_Features).reshape(X_train_seq_len,59,20,1)
X_train_stru = data.loc[:,'ss0':'psa176']
X_train_stru_Features = X_train_stru
X_train_stru_len = len(X_train_stru_Features)
X_train_stru_Features = array(X_train_stru_Features).reshape(X_train_stru_len,59,7,1)
X_train_1D = data.loc[:,'fea1':'fea28']
X_train_1D = array(X_train_1D).reshape(-1,X_train_1D.shape[1],1)
X_train_seq = X_train_seq_Features
X_train_stru = X_train_stru_Features
X_train_1D = X_train_1D

# test data loading
import pandas as pd
import numpy
from numpy import array
data = pd.read_excel('./result/PRE_28fea_59_standardfea_test.xlsx')
y_test = data['Label']
X_test_seq = data.loc[:,'pssm0':'pssm1179']
X_test_seq_Features = X_test_seq
X_test_seq_len = len(X_test_seq_Features)
X_test_seq_Features = array(X_test_seq_Features).reshape(X_test_seq_len,59,20,1)
X_test_stru = data.loc[:,'ss0':'psa176']
X_test_stru_Features = X_test_stru
X_test_stru_len = len(X_test_stru_Features)
X_test_stru_Features = array(X_test_stru_Features).reshape(X_test_stru_len,59,7,1)
X_test_1D = data.loc[:,'fea1':'fea28']
X_test_1D = array(X_test_1D).reshape(-1,X_test_1D.shape[1],1)
X_test_seq = X_test_seq_Features
X_test_stru = X_test_stru_Features
X_test_1D = X_test_1D

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test= to_categorical(y_test)

PRE_28_standardfea = open('./result/PRE_59_27_1-28D-FFMS-Resnet.txt','w')
import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.optimizers import SGD,Adam
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D,Concatenate
from keras.layers import ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation,Input
from keras.layers import Add,GlobalMaxPooling2D
from keras.layers import  LSTM,ConvLSTM2D,concatenate,GlobalMaxPooling2D
from  keras.layers.normalization import BatchNormalization
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D,Add
import keras.backend as K
from keras.layers.core import Lambda
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
channels = 1
nb_classes = 2
batch_size = 32
epochs = 500
filters = 32
kernel_size = 5
pooling_size = 2
color_type = 1

input1=Input(shape=(59, 20, 1))
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

conv21 = Convolution2D(256,(3, 3), strides=(1,1), activation='relu',padding='same')(BN1)
BNor21 = BatchNormalization()(conv21)
MaxPool21 = MaxPooling2D(pool_size=(3,3), strides=(1, 1),border_mode='same')(BNor21)
X21 = convolutional_block(MaxPool21, 3, [256, 256,1024])
X21 = identity_block(X21, 3,[256, 256,1024])
X21 = identity_block(X21, 3, [256, 256,1024])

conv22 = Convolution2D(256,(5, 5), strides=(1,1), activation='relu',padding='same')(BN1)
BNor22 = BatchNormalization()(conv22)
MaxPool22 = MaxPooling2D(pool_size=(5,5), strides=(1, 1),border_mode='same')(BNor22)
X22 = convolutional_block(MaxPool22, 3, [256, 256,1024])
X22 = identity_block(X22, 3, [256, 256,1024])
X22 = identity_block(X22, 3, [256, 256,1024])

conv23 = Convolution2D(256,(7, 7), strides=(1,1), activation='relu',padding='same')(BN1)
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

input2=Input(shape=(59, 7, 1))
conv11s = Convolution2D(128,(3, 3), strides=(1,1), activation='relu',padding='same')(input2)
BNor11s = BatchNormalization()(conv11s)
MaxPool11s = MaxPooling2D(pool_size=(3,3), strides=(1, 1),border_mode='same')(BNor11s)
X11s = convolutional_block(MaxPool11s, 3, [128, 128, 512])
X11s = identity_block(X11s, 3, [128, 128, 512])
X11s = identity_block(X11s, 3, [128, 128, 512])

conv12s = Convolution2D(128,(5, 5), strides=(1,1), activation='relu',padding='same')(input2)
BNor12s = BatchNormalization()(conv12s)
MaxPool12s = MaxPooling2D(pool_size=(5,5), strides=(1, 1),border_mode='same')(BNor12s)
X12s= convolutional_block(MaxPool12s, 3,[128, 128, 512])
X12s = identity_block(X12s, 3, [128, 128, 512])
X12s = identity_block(X12s, 3, [128, 128, 512])

conv13s = Convolution2D(128,(7, 7), strides=(1,1), activation='relu',padding='same')(input2)
BNor13s = BatchNormalization()(conv13s)
MaxPool13s = MaxPooling2D(pool_size=(7,7), strides=(1, 1),border_mode='same')(BNor13s)
X13s = convolutional_block(MaxPool13s, 3, [128, 128, 512])
X13s = identity_block(X13s, 3, [128, 128, 512])
X13s = identity_block(X13s, 3, [128, 128, 512])

add1s =Add()([X11s,X12s,X13s])
BN1s = BatchNormalization()(add1s)
   
conv21s = Convolution2D(256,(3, 3), strides=(1,1), activation='relu',padding='same')(BN1s)
BNor21s = BatchNormalization()(conv21s)
MaxPool21s = MaxPooling2D(pool_size=(3,3), strides=(1, 1),border_mode='same')(BNor21s)
X21s = convolutional_block(MaxPool21s, 3, [256, 256,1024])
X21s = identity_block(X21s, 3,[256, 256,1024])
X21s = identity_block(X21s, 3, [256, 256,1024])

conv22s = Convolution2D(256,(5, 5), strides=(1,1), activation='relu',padding='same')(BN1s)
BNor22s = BatchNormalization()(conv22s)
MaxPool22s = MaxPooling2D(pool_size=(5,5), strides=(1, 1),border_mode='same')(BNor22s)
X22s = convolutional_block(MaxPool22s, 3, [256, 256,1024])
X22s = identity_block(X22s, 3, [256, 256,1024])
X22s = identity_block(X22s, 3, [256, 256,1024])

conv23s = Convolution2D(256,(7, 7), strides=(1,1), activation='relu',padding='same')(BN1s)
BNor23s = BatchNormalization()(conv23s)
MaxPool23s = MaxPooling2D(pool_size=(7,7), strides=(1, 1),border_mode='same')(BNor23s)
X23s = convolutional_block(MaxPool23s, 3, [256, 256,1024])
X23s = identity_block(X23s, 3, [256, 256,1024])
X23s = identity_block(X23s, 3,[256, 256,1024])

add2s =Add()([X21s,X22s,X23s])  
BN2s = BatchNormalization()(add2s)
     
conv31s = Convolution2D(512,(3, 3), strides=(1,1), activation='relu',padding='same')(BN2s)
BNor31s = BatchNormalization()(conv31s)
MaxPool31s = MaxPooling2D(pool_size=(3,3), strides=(1, 1),border_mode='same')(BNor31s)
X31s = convolutional_block(MaxPool31s, 3, [512, 512,2048], s=2)
X31s = identity_block(X31s, 3, [512, 512,2048])
X31s = identity_block(X31s, 3, [512, 512,2048])

conv32s = Convolution2D(512,(5, 5), strides=(1,1), activation='relu',padding='same')(BN2s)
BNor32s = BatchNormalization()(conv32s)
MaxPool32s = MaxPooling2D(pool_size=(5,5), strides=(1, 1),border_mode='same')(BNor32s)
X32s = convolutional_block(MaxPool32s, 3, [512, 512,2048], s=2)
X32s = identity_block(X32s, 3, [512, 512,2048])
X32s = identity_block(X32s, 3, [512, 512,2048])

conv33s = Convolution2D(512,(7, 7), strides=(1,1), activation='relu',padding='same')(BN2s)
BNor33s = BatchNormalization()(conv33s)
MaxPool33s = MaxPooling2D(pool_size=(7,7), strides=(1, 1),border_mode='same')(BNor33s)
X33s = convolutional_block(MaxPool33s, 3, [512, 512,2048], s=2)
X33s = identity_block(X33s, 3, [512, 512,2048])
X33s = identity_block(X33s, 3, [512, 512,2048])

add3s =Add()([X31s,X32s,X33s])
BN3s = BatchNormalization()(add3s)

input_shape=X_train_1D.shape[1:]
input3=Input(input_shape)

conv11D = Conv1D(filters = 128,kernel_size = 3,strides=1,activation='relu', padding='same')(input3)
BNor11D = BatchNormalization()(conv11D)
MaxPool11D = MaxPooling1D(pool_size = 3, strides = 1,border_mode='same')(BNor11D)
X11D = convolutional_block_1D(MaxPool11D, 3, [128, 128, 512])
X11D = identity_block_1D(X11D, 3, [128, 128, 512])
X11D = identity_block_1D(X11D, 3, [128, 128, 512])

conv12D = Conv1D(filters = 128,kernel_size = 5,strides=1,activation='relu', padding='same')(input3)
BNor12D = BatchNormalization()(conv12D)
MaxPool12D = MaxPooling1D(pool_size = 5, strides = 1,border_mode='same')(BNor12D)
X12D = convolutional_block_1D(MaxPool12D, 5,[128, 128, 512])
X12D = identity_block_1D(X12D, 5, [128, 128, 512])
X12D = identity_block_1D(X12D, 5, [128, 128, 512])

conv13D = Conv1D(filters = 128,kernel_size = 7,strides=1,activation='relu', padding='same')(input3)
BNor13D = BatchNormalization()(conv13D)
MaxPool13D = MaxPooling1D(pool_size = 7, strides = 1,border_mode='same')(BNor13D)
X13D = convolutional_block_1D(MaxPool13D, 7, [128, 128, 512])
X13D = identity_block_1D(X13D, 7, [128, 128, 512])
X13D = identity_block_1D(X13D, 7, [128, 128, 512])

add1D =Add()([X11D,X12D,X13D])
BN1D = BatchNormalization()(add1D)

conv21D = Conv1D(filters = 256,kernel_size = 3,strides=1,activation='relu', padding='same')(BN1D)
BNor21D = BatchNormalization()(conv21D)
MaxPool21D = MaxPooling1D(pool_size = 3, strides = 1,border_mode='same')(BNor21D)
X21D = convolutional_block_1D(MaxPool21D, 3, [256, 256,1024])
X21D = identity_block_1D(X21D, 3, [256, 256,1024])
X21D = identity_block_1D(X21D, 3, [256, 256,1024])

conv22D = Conv1D(filters = 256,kernel_size = 5,strides=1,activation='relu', padding='same')(BN1D)
BNor22D = BatchNormalization()(conv22D)
MaxPool22D = MaxPooling1D(pool_size = 5, strides = 1,border_mode='same')(BNor22D)
X22D = convolutional_block_1D(MaxPool22D, 5,[256, 256,1024])
X22D = identity_block_1D(X22D, 5,[256, 256,1024])
X22D = identity_block_1D(X22D, 5,[256, 256,1024])

conv23D = Conv1D(filters = 256,kernel_size = 7,strides=1,activation='relu', padding='same')(BN1D)
BNor23D = BatchNormalization()(conv23D)
MaxPool23D = MaxPooling1D(pool_size = 7, strides = 1,border_mode='same')(BNor23D)
X23D = convolutional_block_1D(MaxPool23D, 7, [256, 256,1024])
X23D = identity_block_1D(X23D, 7, [256, 256,1024])
X23D = identity_block_1D(X23D, 7, [256, 256,1024])

add2D =Add()([X21D,X22D,X23D])
BN2D = BatchNormalization()(add2D)

conv31D = Conv1D(filters = 512,kernel_size = 3,strides=1,activation='relu', padding='same')(BN2D)
BNor31D = BatchNormalization()(conv31D)
MaxPool31D = MaxPooling1D(pool_size = 3, strides = 1,border_mode='same')(BNor31D)
X31D = convolutional_block_1D(MaxPool31D, 3, [512, 512,2048])
X31D = identity_block_1D(X31D, 3, [512, 512,2048])
X31D = identity_block_1D(X31D, 3, [512, 512,2048])

conv32D = Conv1D(filters = 512,kernel_size = 5,strides=1,activation='relu', padding='same')(BN2D)
BNor32D = BatchNormalization()(conv32D)
MaxPool32D = MaxPooling1D(pool_size = 5, strides = 1,border_mode='same')(BNor32D)
X32D = convolutional_block_1D(MaxPool32D, 5,[512, 512,2048])
X32D = identity_block_1D(X32D, 5,[512, 512,2048])
X32D = identity_block_1D(X32D, 5,[512, 512,2048])

conv33D = Conv1D(filters = 512,kernel_size = 7,strides=1,activation='relu', padding='same')(BN2D)
BNor33D = BatchNormalization()(conv33D)
MaxPool33D = MaxPooling1D(pool_size = 7, strides = 1,border_mode='same')(BNor33D)
X33D = convolutional_block_1D(MaxPool33D, 7, [512, 512,2048])
X33D = identity_block_1D(X33D, 7, [512, 512,2048])
X33D = identity_block_1D(X33D, 7, [512, 512,2048])

add3D =Add()([X31D,X32D,X33D])
BN3D = BatchNormalization()(add3D)

inputs=[BN3,BN3s]
g1 = concatenate(inputs, axis=-2)
BN4 = BatchNormalization()(g1)

MaxPool3 = MaxPooling2D(pool_size=(3,3), strides=(1, 1),border_mode='same')(BN4)
X4 = convolutional_block(MaxPool3, 3, [512, 512,2048], s=2)
X4= identity_block(X4, 3, [512, 512, 2048])
X4= identity_block(X4, 3, [512, 512, 2048])

flat1 = Flatten()(X4)
flat2 = Flatten()(BN3D)
g2 = concatenate([flat1,flat2],axis=1)

den1 = Dense(2048, activation='relu')(g2)
drop1 = Dropout(0.4)(den1)
den2 = Dense(1024, activation='relu')(drop1)
drop2 = Dropout(0.4)(den2)
den3 = Dense(512, activation='relu')(drop2)
drop3 = Dropout(0.4)(den3)
den4 = Dense(256, activation='relu')(drop3)
drop4 = Dropout(0.4)(den4)
pred = Dense(nb_classes, activation='softmax')(drop4)
model=Model(inputs=[input1, input2, input3],outputs = pred)
model.summary()

plot_model(model, to_file='./result/FFMSRes-MutP-PSPP-seq.png', show_shapes=True, show_layer_names=False, rankdir='TB', expand_nested=False, style=0, color=True, dpi=200)
SGD = SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=SGD, metrics=['accuracy'])
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3,verbose=1, mode='auto')
hist = model.fit([X_train_seq, X_train_stru, X_train_1D], y_train, batch_size = batch_size,
                 epochs=epochs,verbose=1, validation_split = 0.1, callbacks=[early_stopping])
score = model.evaluate([X_test_seq, X_test_stru, X_test_1D], y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])