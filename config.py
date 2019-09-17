# -*- coding: utf-8 -*-s*
"""
Created on Sat Jul 13 11:59:49 2019

@author: carlo
"""
train_path='Labels/Train'
val_path='Labels/Validation'
test_path='Labels/Test'
model_path='Models/'
log_path = 'log/'
model_name='unet'

num_classes=3
patch_size=(256,256)
epochs=100
steps_per_epoch=1000
validation_steps = 150
frames=1
