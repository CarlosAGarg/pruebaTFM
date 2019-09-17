#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: Carlos A. Garcia Garcia
"""


from Generator import *
from Red import *
from Functions import *
from keras.callbacks import *
from os.path import join, split, normpath, exists
import config as cf

arguments = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

train_path = cf.train_path
val_path = cf.val_path

epochs = cf.epochs
steps_per_epoch = cf.steps_per_epoch
validation_steps=cf.validation_steps
patch_size = cf.patch_size
num_classes= cf.num_classes
model_name = cf.model_name
frames=cf.frames

if not exists(cf.model_path+cf.model_name):
  os.makedirs(cf.model_path+cf.model_name)
elif not exists(cf.log_path+cf.model_name):
  os.makedirs(cf.log_path+cf.model_name)

train_gen = generator(1, train_path, 'image16', 'label',(978, 980), patch_size,arguments, save_to_dir=None, n_classes=num_classes)

val_gen = generator(1, val_path, 'image16', 'label',(978, 980), patch_size ,arguments, save_to_dir=None, n_classes=num_classes)

model = eval(cf.model_name)((patch_size[0], patch_size[1], 1), num_classes)

model_path= "Models/"+model_name+"/model-{epoch:02d}-{val_loss:.5f}.hdf5"
#model_name="8capas10x.hdf5"
callbacks_list=[ModelCheckpoint(model_path, monitor='val_loss',verbose=1, save_best_only=False, mode='min'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
               TensorBoard(log_dir='./log/'+model_name, histogram_freq=0,
                         write_graph=True,
                         write_grads=True,
                         batch_size=1,
                         write_images=True,
                         update_freq=250)]
               
history=model.fit_generator(train_gen,steps_per_epoch=steps_per_epoch,epochs=60,validation_data=val_gen, validation_steps = 150, callbacks=callbacks_list, verbose=1)




