# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 00:20:15 2019

@author: Carlos A
"""
from keras.preprocessing import image
from keras.utils import to_categorical
from Generator import *
from Red import *
import numpy as np
from keras.callbacks import *
from Functions import *
import os
from matplotlib import pyplot as plt
from iterator import *
from utils import *
from Metrics import *
from MultiResUnet3D import *

ruta_app = os.getcwd()
arguments = dict(rotation_range=90,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
#                    vertical_flip=True,
                    fill_mode='nearest')

arguments_val = dict(fill_mode='nearest')

train_path = 'Labels/Train/'
test_images = 'Labels/Test/image/'
test_labels = 'Labels/Test/label/'
val_path = 'Labels/Validation/'
steps_per_epoch = 500
patch_size = (256, 256) 
frames=1
num_classes=3
train_gen = generator(1, train_path, 'image16', 'label',(978, 980), patch_size, arguments=arguments, save_to_dir=None, n_classes=3)

val_gen = generator(1, val_path, 'image16', 'label',(978, 980), patch_size , arguments=arguments_val, save_to_dir=None, n_classes=3)

model = red((patch_size[0], patch_size[1], 1), num_classes)

#☻model_name= "TresCapas10x.hdf5"
callbacks_list=[EarlyStopping(monitor='val_loss',mode='min', patience=2,),
                ModelCheckpoint(model_name, monitor='val_loss',verbose=1, save_best_only=True), 
                ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
                ]


#model_checkpoint = ModelCheckpoint('red.hdf5', monitor='loss',verbose=1, save_best_only=True)
history=model.fit_generator(train_gen,steps_per_epoch=steps_per_epoch,epochs=50, validation_data=val_gen, validation_steps=250, callbacks=callbacks_list, verbose=1)

#model = load_model('model-15-0.00866.hdf5')
[test_images, test_labels] = get_images('Labels/Test/image', 'Labels/Test/label', 25, (980, 978), (256, 256))
#testGene = generator_test(1, test_path,(980, 978), (400, 400) , arguments, save_to_dir=None)
#resultados = model.predict_generator(testGene,8,verbose=2)
test_gen = image_generator(test_images)
resultados = model.predict_generator(test_gen, 25,verbose=1)
res2 = np.uint8(np.argmax(resultados, axis=3))


for i in range(25):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(test_images[i])
    plt.subplot(1, 3, 2)
    plt.imshow(test_labels[i])
    plt.subplot(1, 3, 3)
    plt.imshow(res3[i, :, :])
    
res3 = delete_regions(res2, 30)