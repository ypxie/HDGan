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
    

@autojit
def to_one_hot(indices, maxlen):
    if type(indices) in [int, float]:
       indices = [int(indices)]
    return np.asarray(np.eye(label_indx + 1)[indices])
@autojit
def RGB2GRAY(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
@autojit
def RGB2YUV(input):
    R, G, B= input[...,0],input[...,1],input[...,2]

    Y = (0.299 * R + 0.587 * G + 0.114 * B)
    U = (-0.147 * R + -0.289 * G + 0.436 * B)
    V = (0.615 * R + -0.515 * G + -0.100 * B)
    return np.stack([Y, U, V], axis = -1).astype(int)
@autojit
def YUV2RGB(input):
    Y, U, V= input[...,0],input[...,1],input[...,2]
    R = (Y + 1.14 * V)
    G = (Y - 0.39 * U - 0.58 * V)
    B = (Y + 2.03 * U)
    return np.stack([R, G, B], axis = -1).astype(int)

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

def mysqueeze(a, axis = None):
    if axis == None:
        return np.squeeze(a)
    if a.shape[axis] != 1:
        return a
    else:
        return np.squeeze(a, axis = axis)




def imshow(img, size=None):
    if size is not None:
        plt.figure(figsize = size)
    else:
        plt.figure()
    plt.imshow(img)
    plt.show()



def patchflow(Img,chunknum,row,col,channel,**kwargs):

    pixelind = find(np.ones(Img.shape[0], Img.shape[1]) == 1)
    Totalnum = len(pixelind)
    numberofchunk = np.floor((Totalnum + chunknum - 1)// chunknum)   # the floor
    Chunkfile = np.zeros((chunknum, row*col*channel))

    chunkstart = 0
    for chunkidx in range(numberofchunk):
        thisnum = min(chunknum, Totalnum - chunkidx*chunknum)
        thisInd = pixelind[chunkstart: chunkstart + thisnum]
        fast_Points2Patches(Chunkfile[0:thisnum,:],thisInd, Img, (row,col))
        chunkstart += thisnum
        yield Chunkfile[0:thisnum,:]

def Indexflow(Totalnum, batch_size, random=True):
    numberofchunk = int(Totalnum + batch_size - 1)// int(batch_size)   # the floor
    #Chunkfile = np.zeros((batch_size, row*col*channel))
    totalIndx = np.arange(Totalnum).astype(np.int)
    if random is True:
        totalIndx = np.random.permutation(totalIndx)
        
    chunkstart = 0
    for chunkidx in range(int(numberofchunk)):
        thisnum = min(batch_size, Totalnum - chunkidx*batch_size)
        thisInd = totalIndx[chunkstart: chunkstart + thisnum]
        chunkstart += thisnum
        yield thisInd

def IndexH5(h5_array, indices):
    read_list = []
    for idx in indices:
        read_list.append(h5_array[idx])
    return np.stack(read_list, 0)



def batchflow(batch_size, *Data):
    # we dont check Data, they should all have equal first dimension
    Totalnum = Data[0].shape[0]
    for thisInd in Indexflow(Totalnum, batch_size):
        if len(Data) == 1:
            yield Data[0][thisInd, ...]
        else:
            batch_tuple = [s[thisInd,...] for s in Data]
            yield tuple(batch_tuple)
