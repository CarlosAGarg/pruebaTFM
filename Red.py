# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 00:33:56 2019

@author: Carlos A
"""

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from Metrics import *

def red(input_size, num_classes):
    inputs = Input(input_size)
    conv1 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_0", use_bias=True, activation="relu")(inputs)
    conv2 = Conv2D(512, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_1", use_bias=True, activation="relu")(conv1) 
    conv3 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_2", use_bias=True, activation="relu")(conv2) 
    out = Conv2D(3, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="out", use_bias=True, activation="softmax")(conv3)
    model = Model(inputs=inputs, outputs=out)    
    model.compile(optimizer=Adam(lr=1e-4),loss="categorical_crossentropy", metrics=["accuracy", mean_iou])     
    
    print('CNN initialized')
    return model

def red8Capas(input_size, num_classes):
    inputs = Input(input_size)
    conv1 = Conv2D(128, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_0", use_bias=True, activation="relu")(inputs)
    conv2 = Conv2D(128, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_1", use_bias=True, activation="relu")(conv1)
    conv3 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_2", use_bias=True, activation="relu")(conv2)
    conv4 = Conv2D(512, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_3", use_bias=True, activation="relu")(conv3)
    conv5 = Conv2D(512, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_4", use_bias=True, activation="relu")(conv4)
    conv6 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_5", use_bias=True, activation="relu")(conv5)
    conv7 = Conv2D(128, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_6", use_bias=True, activation="relu")(conv6)
    conv8 = Conv2D(128, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_7", use_bias=True, activation="relu")(conv7)
    out = Conv2D(3, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="out", use_bias=True, activation="softmax")(conv8)  
    model = Model(input = inputs, output = out)
    if num_classes==2:
      loss='binary_crossentropy'
    else:
      loss='categorical_crossentropy'
    model.compile(optimizer=Adam(lr=1e-4),loss=loss, metrics=["accuracy", Mean_IOU])     
    
    print('CNN initialized')
    return model
  
  
def redBN(input_size, num_classes):
    inputs = Input(input_size)
    inputs = BatchNormalization()(inputs)
    conv1 = Conv2D(128, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_0", use_bias=True, activation="relu")(inputs)
    conv2 = Conv2D(128, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_1", use_bias=True)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    conv3 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_2", use_bias=True, activation="relu")(conv2)
    conv4 = Conv2D(512, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_3", use_bias=True, activation="relu")(conv3)
    conv5 = Conv2D(512, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_4", use_bias=True, activation="relu")(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    conv6 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_5", use_bias=True, activation="relu")(conv5)
    conv7 = Conv2D(128, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_6", use_bias=True, activation="relu")(conv6)
    conv8 = Conv2D(128, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_7", use_bias=True, activation="relu")(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv5)
    out = Conv2D(3, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="out", use_bias=True, activation="softmax")(conv8)  
    model = Model(input = inputs, output = out)
    model.compile(optimizer=Adam(lr=1e-4),loss=jaccard_distance, metrics=["accuracy"])     
    
    print('CNN initialized')
    return model

def redRes(input_size, num_classes):
    inputs = Input(input_size)
    conv1 = Conv2D(128, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_0", use_bias=True, activation="relu")(inputs)
    conv2 = Conv2D(128, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_1", use_bias=True, activation="relu")(conv1)
    conv3 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_2", use_bias=True, activation="relu")(conv2)
    conv3_shortcut = conv3
    conv4 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_3", use_bias=True, activation="relu")(conv3)
    conv5 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_4", use_bias=True, activation="relu")(conv4)
    conv6 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_5", use_bias=True)(conv5)
    conv6 = Add()([conv6, conv3_shortcut])
    conv6 = Activation('relu')(conv6)
    conv7 = Conv2D(512, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_6", use_bias=True, activation="relu")(conv6)
    conv7_shortcut = conv7
    conv8 = Conv2D(512, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_7", use_bias=True, activation="relu")(conv7)
    conv9 = Conv2D(512, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_8", use_bias=True, activation="relu")(conv8)
    conv10 = Conv2D(512, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_9", use_bias=True)(conv9)
    conv10 = Add()([conv10, conv7_shortcut])
    conv10 = Activation("relu")(conv10)
    conv11 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_10", use_bias=True, activation="relu")(conv10)
    conv11_shortcut = conv11
    conv12 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_11", use_bias=True, activation="relu")(conv11)
    conv13 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_12", use_bias=True, activation="relu")(conv12)
    conv14 = Conv2D(256, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_13", use_bias=True)(conv13)
    conv14 = Add()([conv14, conv11_shortcut])
    conv14= Activation("relu")(conv14)
    conv15 = Conv2D(128, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_14", use_bias=True, activation="relu")(conv14)
    conv16 = Conv2D(128, (3,3), strides=(1,1), padding="same", kernel_initializer="he_normal", name="conv2d_15", use_bias=True, activation="relu")(conv15)
    out = Conv2D(3, (1,1), strides=(1,1), padding="same", kernel_initializer="he_normal", name="out", use_bias=True, activation="softmax")(conv16)  
    model = Model(input = inputs, output = out)
    if num_classes==2:
      loss='binary_crossentropy'
    else:
      loss='categorical_crossentropy'
    model.compile(optimizer=Adam(lr=1e-4),loss=loss, metrics=["accuracy", Mean_IOU])     
    
    print('CNN initialized')
    return model
  
def unet(input_size, num_classes):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(num_classes, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    if num_classes==2:
      loss='binary_crossentropy'
    else:
      loss='categorical_crossentropy'
    model.compile(optimizer = Adam(lr = 1e-4), loss = loss, metrics = ['accuracy', mean_iou]) 
    return model

from keras.models import Model
from keras.layers import Activation,Input,ZeroPadding2D,Cropping2D
from custom_layers import MaxPoolingWithIndices,UpSamplingWithIndices,CompositeConv

def segnet(input_size, num_classes):
    inputs=Input(input_size)

    x = ZeroPadding2D(((8,8),(16,16)))(inputs)

    x=CompositeConv(x,2,64)
    x,argmax1=MaxPoolingWithIndices(pool_size=2,strides=2)(x)
    
    x=CompositeConv(x,2,64)
    x,argmax2=MaxPoolingWithIndices(pool_size=2,strides=2)(x)
    
    x=CompositeConv(x,3,64)
    x,argmax3=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=CompositeConv(x,3,64)
    x,argmax4=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=CompositeConv(x,3,64)
    x,argmax5=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=UpSamplingWithIndices()([x,argmax5])
    x=CompositeConv(x,3,64)

    x=UpSamplingWithIndices()([x,argmax4])
    x=CompositeConv(x,3,64)

    x=UpSamplingWithIndices()([x,argmax3])
    x=CompositeConv(x,3,64)

    x=UpSamplingWithIndices()([x,argmax2])
    x=CompositeConv(x,2,64)
    
    x=UpSamplingWithIndices()([x,argmax1])
    x=CompositeConv(x,2,[64,num_classes])

    x=Activation('softmax')(x)

    y=Cropping2D(((8,8),(16,16)))(x)
    
    model=Model(inputs=inputs,outputs=y)
    if num_classes==2:
      loss='binary_crossentropy'
    else:
      loss='categorical_crossentropy'
    model.compile(optimizer=Adam(lr=1e-4),loss=loss, metrics=["accuracy", Mean_IOU])     
    
    print('CNN initialized')
    return model





