# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:59:05 2019

@author: carlo
"""
'''
from tensorflow.keras.utils.data_utils import Sequence
import os
import SimpleITK as sitk


class CIFAR10Sequence(Sequence):
    def __init__(self, image_path, label_path, batch_size, augmentations):
        x_set=[]
        y_set=[]
        os.chdir(image_path)
        contenido = os.listdir(image_path)
        for i in range(len(contenido)):
            imagen = sitk.ReadImage(contenido[i])
            imagen = sitk.GetArrayFromImage(imagen)
            x_set.append(imagen)
            
        os.chdir('..\\..\\..\\')
        os.chdir(label_path)
        contenido = os.listdir(label_path)
        for j in range(len(contenido)):
            label = sitk.ReadImage(contenido[i])
            label = sitk.GetArrayFromImage(label)
            y_set.append(label)

        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        return np.stack([
            self.augment(image=x)["image"] for x in batch_x
        ], axis=0), np.array(batch_y)
    
    
import cv2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, ShiftScaleRotate
)

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    RandomContrast(limit=0.2, p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    RandomBrightness(limit=0.2, p=0.5),
    HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
                       val_shift_limit=10, p=.9),
    # CLAHE(p=1.0, clip_limit=2.0),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1, 
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8), 
    ToFloat(max_value=255)
])

AUGMENTATIONS_TEST = Compose([
    # CLAHE(p=1.0, clip_limit=2.0),
    ToFloat(max_value=255)
])
'''
import Augmentor

p = Augmentor.Pipeline("Labels/Train/image16")





























