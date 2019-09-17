# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:56:00 2019

@author: Carlos A
"""
from skimage import io
import os
from skimage.io import imsave
import numpy as np
import SimpleITK as sitk

ruta_app = os.getcwd()
origin = ruta_app+'\\Labels\\GuardadoLabel'
train= ruta_app+'\\Labels\\label'
test=ruta_app+'\\Labels\\Test\\label'
validation=ruta_app+'\\Labels\\Validation\\label'
os.chdir(origin)
contenido = os.listdir(ruta_app)
for i in range(len(contenido)):
    os.chdir(ruta_app)
    segmentacion = sitk.ReadImage(contenido[i])
    segmentacion = sitk.GetArrayFromImage(segmentacion)
    frames = np.argsort(segmentacion.shape)[0]
    print(segmentacion.shape)
    print(frames)
    for j in range(segmentacion.shape[frames]):
        if frames==0:
            aux = (segmentacion[j, :, :])
        elif frames==2:
            aux = (segmentacion[:, :, j])
        #aux = morphology.closing(aux)
        #print(aux.shape)
        #print(aux.dtype)
        #aux=morphology.remove_small_objects(aux, 30)
        #print(aux.shape)
        #print(aux.dtype)
        #aux=morphology.remove_small_holes(aux, 15,in_place=True)
        #print(aux.shape)
        #print(aux.dtype)
        if aux.shape[1]==982:
            aux = aux[3:-2, 0:-2]
        else:
            aux = aux[3:-2, 3:-2]
        imsave(str(j)+'_'+contenido[i], aux, plugin='tifffile')


#Saving Evaluation Images
ruta_app = os.getcwd()
ruta_origen = "Labels\\Test\\label"
ruta_destino = "EvaluationDataset\\01_GT\\SEG\\"
contenido = os.listdir(ruta_origen)
for i in range(len(contenido)):
    os.chdir(ruta_origen)
    segmentacion = sitk.ReadImage(contenido[i])
    segmentacion = sitk.GetArrayFromImage(segmentacion)
    os.chdir(ruta_app)
    os.chdir(ruta_destino)
    imsave("man_seg"+'%03d' % int(i)+"tif", segmentacion, plugin='tifffile')


ruta_app = os.getcwd()
train= ruta_app+'\\Labels\\label'
test=ruta_app+'\\Labels\\Test\\label'
validation=ruta_app+'\\Labels\\Validation\\label'
contenido = os.listdir(ruta_app)
for i in range(len(contenido)):
    os.chdir(ruta_app)
    segmentacion = sitk.ReadImage(contenido[i])
    segmentacion = sitk.GetArrayFromImage(segmentacion)
    segmentacion = np.int16(segmentacion > 0)
    os.chdir('..\\labelBinaria')
    imsave(contenido[i], segmentacion, plugin='tifffile')
    
    
    
    
ruta_app = os.getcwd()
contenido = os.listdir(ruta_app)
for i in range(len(contenido)):
    os.chdir(ruta_app)
    segmentacion = sitk.ReadImage(contenido[i])
    segmentacion = sitk.GetArrayFromImage(segmentacion)
    #if segmentacion.shape[0]==985:
    segmentacion = segmentacion[0:-1, 0:-2]
    #else:
     #   segmentacion = segmentacion[0:-2, 3:-2]
    print(segmentacion.shape)
    imsave(contenido[i], segmentacion, plugin='tifffile')