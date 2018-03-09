# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import os, math
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import scipy
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import skimage, skimage.morphology
import cv2
from PIL import Image, ImageDraw
from scipy.ndimage.interpolation import rotate
from skimage import color, measure
import re
import scipy.ndimage

from numba import jit, autojit

import random, shutil
import scipy.misc as misc


def mkdirs(folders, erase=False):
    if type(folders) is not list:
        folders = [folders]
    for fold in folders:
        if not os.path.exists(fold):
            os.makedirs(fold)
        else:
            if erase:
                shutil.rmtree(fold)
                os.makedirs(fold)
                
def normalize_img(X):
    min_, max_ = np.min(X), np.max(X)
    X = (X - min_)/ (max_ - min_ + 1e-9)
    X = X*255
    return X.astype(np.uint8)
    
def imread(imgfile):
    assert os.path.exists(imgfile), '{} does not exist!'.format(imgfile)
    srcBGR = cv2.imread(imgfile)
    destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
    return destRGB

def writeImg(array, savepath):      
    #scipy.misc.imsave(savepath, array)
    cv2.imwrite(savepath,  array)
    

def imresize(img, resizeratio=1):
    '''Take care of cv2 reshape squeeze behevaior'''
    if resizeratio == 1:
        return img
    outshape = ( int(img.shape[1] * resizeratio) , int(img.shape[0] * resizeratio))
    # temp = cv2.resize(img, outshape).astype(float)
    temp = misc.imresize(img, size=outshape).astype(float)
    if len(img.shape) == 3 and img.shape[2] == 1:
        temp = np.reshape(temp, temp.shape + (1,))
    return temp


def imresize_shape(img, outshape):
    if len(img.shape) == 3:
        if img.shape[0] == 1 or img.shape[0] == 3:
            transpose_img = np.transpose(img, (1,2,0))
            _img =  imresize_shape(transpose_img, outshape)
            return np.transpose(_img, (2,0, 1))
    if len(img.shape) == 4: 
        img_out = []
        for this_img in img:
            img_out.append( imresize_shape(this_img, outshape) ) 
        return np.stack(img_out, axis=0)

    img = img.astype(np.float32)
    outshape = ( int(outshape[1]) , int(outshape[0])  )
    
    #temp = cv2.resize(img, outshape).astype(float)
    temp = misc.imresize(img, size=outshape, interp='bilinear').astype(float)

    if len(img.shape) == 3 and img.shape[2] == 1:
        temp = np.reshape(temp, temp.shape + (1,))
    return temp


def imshow(img, size=None):
    if size is not None:
        plt.figure(figsize = size)
    else:
        plt.figure()
    plt.imshow(img)
    plt.show()

