# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:19:47 2019

@author: Carlos A
"""

from skimage import io
from skimage.io import imsave
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
#!pip install SimpleITK
import SimpleITK as sitk
from os.path import join, split, normpath, exists

def get_patches(img, mask, width, height):
    #Binary mask
    #img_aux = img[width/2:-width/2, height/2:-height/2]
    #print(mask.shape)
    aux = np.zeros(((img.shape[0]+(width/2)), img.shape[0]+(height/2)))
    mask_aux = mask[int(width/2):int(-width/2), int(height/2):int(-height/2)]
    #print(mask_aux.shape)
    bin_mask = mask_aux > 0
    bin_mask = np.float32(bin_mask)
    aux = np.zeros(bin_mask.shape)
    aux = aux + 0.0001 #0.01
    bin_mask = bin_mask + aux
    pdf = bin_mask.ravel()/np.sum(bin_mask)
    choices = np.prod(pdf.shape)
    x = img.shape[0]
    y = img.shape[1]
    i=0
    x=-1
    y=-1
    while x<0 or y<0:
        i=i+1
        index = np.random.choice(choices, size=1, p=pdf)
        coords = np.array(np.unravel_index(index, shape=bin_mask.shape)).flatten()
        x = np.int(coords[1] - (width/2))
        y = np.int(coords[0] - (height/2))
        #x = np.int(coords[1])
        #y = np.int(coords[0])
        if i > 5:
            x=0
            y=0
    img = img[x:x+width, y:y+height]
    mask = mask[x:x+width, y:y+height]
    #print(x)
    #print(y)
    img = img.reshape((1,)+(img.shape))
    mask = mask.reshape((1,)+(mask.shape))
    return img, mask

def save_results(gt, seg, path_gt, path_seg):
    if not exists(path_gt):
        os.makedirs(path_gt)
    elif not exists(path_seg):
        os.makedirs(path_seg)
    for i in range(seg.shape[0]):
        os.chdir(path_seg)
        image = seg[i, :, :]
        image = np.uint16(image)
        imsave("mask"+'%03d' % int(i)+".tif", image, plugin='tifffile')
        os.chdir("../../")
        os.chdir(path_gt)
        label = gt[i, :, :]
        label = np.uint16(label)
        imsave("man_seg"+'%03d' % int(i)+".tif", label, plugin='tifffile')
        os.chdir("../../../")
        
        
def get_images(path_image, path_label, num_image, target_size, patch_size):
    patch_width = patch_size[0]
    patch_height = patch_size[1]
    contenido_im = os.listdir(path_image)
    contenido_label = os.listdir(path_label)
    contenido_label.sort()
    contenido_im.sort()
    os.chdir(path_label)
    labels = []
    test_images=[]
    test_labels=[]
    for i in range(num_image):
        mask = sitk.ReadImage(contenido_label[i])
        mask = sitk.GetArrayFromImage(mask)
        labels.append(mask)
    os.chdir('../../../')
    os.chdir(path_image)
    images = []
    for j in range(num_image):
        img = sitk.ReadImage(contenido_im[i])
        img = sitk.GetArrayFromImage(img)
        img = img/np.max(img)
        [img, mask] = get_patches(img, labels[i], patch_width, patch_height)
        test_images.append(img[0, :, :])
        test_labels.append(mask[0, :, :]) 
    return test_images, test_labels

from skimage import measure
def delete_regions(images, threshold):
    for i in range(len(images)):
        labeled = measure.label(images[i, :, :])
        props = measure.regionprops(labeled)
        for j in range(len(props)):
            if props[j].area < threshold:
                images[i, props[j].coords]=0
    return images

from skimage import morphology
def postproc(images, threshold):
    es = np.ones((3,3))
    for i in range(len(images)):
        im = images[i, :, :]
        images[i, :, :] = morphology.closing(im, es)
        #labeled = measure.label(images[i, :, :], neighbors=8)
        props = measure.regionprops(np.uint8(images[i, :, :]))
        for j in range(len(props)):
            if props[j].area < threshold:
                images[i, props[j].coords]=0
    return images




from sklearn.feature_extraction.image import *
def predict_whole_images(image_path, model, patch_size, file_extension, n_images):
    pims_sequence = pims.ImageSequence(join(image_path, '*.{}'.format(file_extension)), process_func=None)
    images = np.stack([frame.copy() for frame in pims_sequence], axis=2)
    outputs=[]
    inputs=[]
    for j in range(n_images):
        aux=images[:, :, j]
        aux = resize_image(aux, [768, 768])
        inputs.append(aux)
        patches = split_image(aux, (256,256))
        patches = patches.reshape(patches.shape+(1,))
        resultado = model.predict(patches, verbose=2)
        resultado = np.uint8(np.argmax(resultado, axis=3))
        imagen = reconstruction(resultado, (768, 768))
        outputs.append(imagen)
    return outputs, inputs

def resize_image(image, size):
    """
    Crop the image with a centered rectangle of the specified size
    image:      a Pillow image instance
    size:       a list of two integers [width, height]
    """
    img_format = image.dtype
    image = image.copy()
    old_size = image.shape
    left = (old_size[0] - size[0]) / 2
    top = (old_size[1] - size[1]) / 2
    right = old_size[0] - left
    bottom = old_size[1] - top
    rect = [int(np.math.ceil(x)) for x in (left, top, right, bottom)]
    left, top, right, bottom = rect
    crop = image[top:bottom, left:right]
    crop.dtype = img_format
    return crop

import pims
from os.path import join, split, normpath, exists
def read_groundtruth_sequence(path, file_extension):
    pims_sequence = pims.ImageSequence(join(path, '*.{}'.format(file_extension)), process_func=None)
    return np.stack([resize_image(frame.copy(), (768,768)) for frame in pims_sequence], axis=2)

def convert_to_gif(video, name):
    image=[]
    frames = np.argsort(video.shape)[0]
    for j in range(video.shape[frames]):
            if frames==0:
                aux = video[j, :, :]
            elif frames==2:
                aux = video[:, :, j]
            image.append(aux)
    imageio.mimsave(name+'.gif', image)
    

def reconstruction(patches, image_size):
    final = np.zeros(image_size)
    final[0:256, 0:256] = patches[0, :, :]
    final[0:256, 256:512] = patches[1, :, :]
    final[0:256, 512:768] = patches[2, :, :]
    final[256:512, 0:256] = patches[3, :, :]
    final[256:512, 256:512] = patches[4, :, :]
    final[256:512, 512:768] = patches[5, :, :]
    final[512:768, 0:256] = patches[6, :, :]
    final[512:768, 256:512] = patches[7, :, :]
    final[512:768, 512:768] = patches[8, :, :]
    return final

def split_image(image, patch_size):
    patches = np.zeros((9, patch_size[0], patch_size[1]))
    patches[0, :, :] = image[0:256, 0:256]
    patches[1, :, :] = image[0:256, 256:512]
    patches[2, :, :] = image[0:256, 512:768]
    patches[3, :, :] = image[256:512, 0:256]
    patches[4, :, :] = image[256:512, 256:512]
    patches[5, :, :] = image[256:512, 512:768]
    patches[6, :, :] = image[512:768, 0:256]
    patches[7, :, :] = image[512:768, 256:512]
    patches[8, :, :] = image[512:768, 512:768]
    return patches

def predict_videos(path, num_videos):
    videos = os.listdir(join(path))
    pred_videos=[]
    for i in range(num_videos):
        pims_sequence = pims.TiffStack(join(path, videos[i]), process_func=None)
        frames = np.stack([frame.copy() for frame in pims_sequence], axis=2)
        pred_frames=np.zeros((768,768,frames.shape[2]))
        for j in range(frames.shape[2]):
            aux=frames[:, :, j]
            aux = resize_image(aux, [768, 768])
            patches = split_image(aux, (256,256))
            patches = patches.reshape(patches.shape+(1,))
            resultado = model.predict(patches, verbose=2)
            resultado = np.uint8(np.argmax(resultado, axis=3))
            imagen = reconstruction(resultado, (768, 768))
            pred_frames[:, :, j] = imagen
        pred_videos.append(pred_frames)
    return pred_videos
    