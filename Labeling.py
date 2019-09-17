# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:28:01 2019

@author: Carlos A
"""

from skimage import io
import os
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage.io import imsave
import SimpleITK as sitk

ruta_app = os.getcwd()

'''contenido_pred = os.listdir(ruta_app+'\\PREDICCION_3DUNET')

os.chdir(ruta_app+'\\PREDICCION_3DUNET')  
#for i in range(len(contenido_pred)):
#    prediccion = io.imread(contenido_pred[i])
    
contenido_seg = os.listdir(ruta_app+'\\SEGMENTACIONES_MANUALES\\OUTPUT')
#for i in range(len(contenido_seg)):
for i in range(len(contenido_seg)):
    os.chdir(ruta_app+'\\SEGMENTACIONES_MANUALES\\OUTPUT');
    #segmentacion = io.imread(contenido_seg[i])[:, 3:-2, 3:-2]
    segmentacion = io.imread(contenido_seg[i])
    print(segmentacion.shape)
    if segmentacion.shape[0] in range(980, 985, 1):
        #segmentacion = np.swapaxes(segmentacion, 0, 1)
        seg = np.zeros((3, 983, 985))
        seg[0, :, :] = segmentacion[:, :, 0]
        seg[1, :, :] = segmentacion[:, :, 1]
        seg[2, :, :] = segmentacion[:, :, 2]
        print('MIRAAAA')
        print(seg.shape)
        segmentacion=seg
        segmentacion = segmentacion[:, 3:-2, 3:-2]
        print(segmentacion.shape)
        print('LOOOOOOOOOOOK')
    else:
        
        segmentacion = segmentacion[:, 3:-2, 3:-2]
        
    os.chdir(ruta_app+'\\SEGMENTACIONES_MANUALES2\\OUTPUT');
    print(contenido_seg[i])
    print(segmentacion.shape)
    segmentacion=sitk.GetImageFromArray(np.uint8(segmentacion))
    sitk.WriteImage(segmentacion, contenido_seg[i])
    #imsave(contenido_seg[i], segmentacion, plugin="tifffile")
    
print('--------->INPUTS <--------------')
contenido_seg = os.listdir(ruta_app+'\\SEGMENTACIONES_MANUALES\\INPUT')
#for i in range(len(contenido_seg)):
for i in range(len(contenido_seg)):
    os.chdir(ruta_app+'\\SEGMENTACIONES_MANUALES\\INPUT');
    #segmentacion = io.imread(contenido_seg[i])[:, 3:-2, 3:-2]
    segmentacion = io.imread(contenido_seg[i])
    print(segmentacion.shape)
    if segmentacion.shape[0] in range(980, 985, 1):
        #segmentacion = np.swapaxes(segmentacion, 0, 1)
        seg = np.zeros((3, 983, 985))
        seg[0, :, :] = segmentacion[:, :, 0]
        seg[1, :, :] = segmentacion[:, :, 1]
        seg[2, :, :] = segmentacion[:, :, 2]
        print('MIRAAAA')
        print(seg.shape)
        segmentacion=seg
        segmentacion = segmentacion[:, 3:-2, 3:-2]
        print(segmentacion.shape)
        print('LOOOOOOOOOOOK')
    else:
        
        segmentacion = segmentacion[:, 3:-2, 3:-2]
        
    os.chdir(ruta_app+'\\SEGMENTACIONES_MANUALES2\\INPUT');
    print(contenido_seg[i])
    print(segmentacion.shape)
    segmentacion=sitk.GetImageFromArray(np.uint16(segmentacion))
    sitk.WriteImage(segmentacion, contenido_seg[i])'''
    
    