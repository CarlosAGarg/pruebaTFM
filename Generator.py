# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 00:03:43 2019

@author: Carlos A
"""
#from model import *
from keras.preprocessing.image import ImageDataGenerator
#from ImageDataGenerator import *
from keras.preprocessing import image
from matplotlib import pyplot as plt 
from keras.preprocessing import image
from sklearn.feature_extraction import image as im
import os
import glob
import skimage.io as io
import skimage.transform as trans
from sklearn.feature_extraction.image import extract_patches_2d
from keras.utils import to_categorical
import numpy as np
from Functions import *
import SimpleITK as sitk

def generator(batch_size, train_path, images, masks, image_size, patch_size, arguments, save_to_dir, n_classes):
    
    im_gen = ImageDataGenerator(**arguments)
    
    mask_gen=ImageDataGenerator(**arguments)
    
    patch_width = patch_size[0]
    patch_height = patch_size[1]
    
    im_generator = im_gen.flow_from_directory(train_path, 
                                              classes = [images], 
                                              class_mode=None,
                                              color_mode='grayscale',
                                              target_size=image_size, 
#                                              target_size=None,
                                              batch_size=batch_size,
                                              save_to_dir=save_to_dir,
                                              seed = 1)
    mask_generator = mask_gen.flow_from_directory(train_path,
                                              classes = [masks], 
                                              class_mode=None,
                                              color_mode='grayscale',
                                              target_size=image_size,
#                                              target_size=None,
                                              batch_size=batch_size,
                                              save_to_dir=save_to_dir,
                                              seed = 1)
    train_generator = zip(im_generator, mask_generator)
    for (img, mask) in train_generator:
        #print('ENTRADAAA GENERADOOOOOOOR')
        #print(img.shape)
        #print(mask.shape)
        img = img[0, :, :, 0]
        #print(np.min(img))
        #print(np.max(img))
        #print(img.dtype)
        #plt.figure()
        #plt.imshow(img)
        img = img/np.max(img)
        #print(np.min(img))
        #print(np.max(img))
        #print(img.dtype)
        #plt.figure()
        #plt.imshow(img)
#        plt.show()
        #img = extract_patches_2d(img, (patch_size), 1, 1)
        #aux = img[0, :, :]
        #print(aux.shape)
        #plt.imshow(aux)
        #img = img.reshape((img.shape)+(1,))
        mask = mask[0, :, :, 0]
        #print(mask.dtype)
#        plt.figure()
#        plt.imshow(mask)
#        plt.show()
        #mask = extract_patches_2d(mask, (patch_size), 1, 1)
        [img, mask] = get_patches(img, mask, patch_width, patch_height)
        aux2=img[0, :, :]
        #print(aux2.shape)
        aux3=mask[0, :, :]
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(aux2)
        plt.gca().set_title('Input')
        plt.subplot(1, 2, 2)
        plt.imshow(aux3)
        plt.gca().set_title('Output')
        plt.show()
        img = img.reshape((img.shape)+(1,))
        mask = to_categorical(mask, n_classes)
        mask = mask.astype('int8')
        img = img.reshape((batch_size,patch_width,patch_height,1))
        mask = mask.reshape((batch_size,patch_width,patch_height,n_classes))
        #print(mask.dtype)
        #mask = mask.reshape((mask.shape)+(1,))
        #print('SALIDAAAAA GENERADOOOOOOOR')
        #print(img.shape)
        #print(mask.shape)
        #print(img.dtype)
        yield(img, mask)
        

def test_generator(path_image, path_label, num_image, target_size, patch_size):
    #target_size = (256,256)
    patch_width = patch_size[0]
    patch_height = patch_size[1]
    contenido_im = os.listdir(path_image)
    contenido_label = os.listdir(path_label)
    os.chdir(path_label)
    labels = []
    test_images=[]
    test_labels=[]
    for i in range(num_image):
        #mask = io.imread(contenido_label[i],as_gray = True)
        #mask = trans.resize(mask,target_size)
        mask = sitk.ReadImage(contenido_label[i])
        mask = sitk.GetArrayFromImage(mask)
        labels.append(mask)
    os.chdir('../../../')
    os.chdir(path_image)
    images = []
    for i in range(num_image):
        #img = io.imread(contenido_im[i],as_gray = True)
        img = sitk.ReadImage(contenido_im[i])
        img = sitk.GetArrayFromImage(img)
        #print(np.max(img))
        #print('ENTRADAAA GENERADOOOOOOOR')
        #print(img.shape)
        img = img/np.max(img)
        img = trans.resize(img,target_size)
        [img, mask] = get_patches(img, labels[i], patch_width, patch_height)
        test_images.append(img[0, :, :])
        test_labels.append(mask[0, :, :])
        #aux2=img[0, :, :]
        #print(mask.shape)
        #plt.figure()
        #plt.imshow(aux2)
        #aux3=mask[0, :, :]
        #print(mask.shape)
        #plt.figure()
        #plt.imshow(aux3)
        #img = extract_patches_2d(img, (patch_size), 1)
        img = img.reshape((img.shape)+(1,))
        img = np.float32(img)
        img = img/np.max(img)
        #print('SALIDAAAAA GENERADOOOOOOOR')
        #print(img.shape)
        yield img
        return test_images, test_labels
        
        
def predict_generator(batch_size, train_path, images, masks, image_size, patch_size, arguments, save_to_dir):
    patch_width = patch_size[0]
    patch_height = patch_size[1]
    im_gen = ImageDataGenerator(**arguments)
    mask_gen=ImageDataGenerator(**arguments)
    im_generator = im_gen.flow_from_directory(train_path,  
                                              classes=[images],
                                              class_mode=None,
                                              color_mode='grayscale',
                                              target_size=image_size, 
                                              batch_size=batch_size,
                                              save_to_dir=save_to_dir,
                                              seed = 1)
    mask_generator = mask_gen.flow_from_directory(train_path,
                                              classes = [masks], 
                                              class_mode=None,
                                              color_mode='grayscale',
                                              target_size=image_size, 
                                              batch_size=batch_size,
                                              save_to_dir=save_to_dir,
                                              seed = 1)
    train_generator = zip(im_generator, mask_generator)
    for img,mask in train_generator:
        print('ENTRADAAA GENERADOOOOOOOR')
        print(img.shape)
        img = img.reshape(image_size)
        #print(np.min(img))
        #print(np.max(img))
        #print(img.dtype)
        img = img/np.max(img)
        #print(np.min(img))
        #print(np.max(img))
        #print(img.dtype)
        #plt.imshow(img, cmap='gray')
        [img, mask] = get_patches(img, mask, patch_width, patch_height)
        #aux = img[0, :, :]
        #print(aux.shape)
        #plt.imshow(aux)
        img = img.reshape((img.shape)+(1,))
        yield(img)
        
def image_generator(test_images):
    
    for i in range(len(test_images)):
        img = test_images[i]
        img = img.reshape((img.shape)+(1,))
        img =img.reshape((1,)+(img.shape))
        yield img