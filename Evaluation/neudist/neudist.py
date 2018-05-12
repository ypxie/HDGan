# -*- coding: utf-8 -*-

import os
import sys, os
sys.path.insert(0, os.path.join('..'))
proj_root = os.path.join('..', '..')
data_root = os.path.join(proj_root, 'Data')
model_root = os.path.join(proj_root, 'Models')
import argparse
import torch, h5py
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from HDGan.neuralDist.testNeuralDist  import test_nd
from HDGan.neuralDist.neuralDistModel  import ImgSenRanking
from HDGan.neuralDist.neuralDistModel  import ImageEncoder


if  __name__ == '__main__':
    
    dim_image    =  1536
    sent_dim     =  1024
    hid_dim      =  512

    parser = argparse.ArgumentParser(description = 'test nd') 
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='batch size.')
    
    parser.add_argument('--device_id', type=int, default=0, 
                        help='which device')
    parser.add_argument('--dataset',    type=str,      default= None, 
                        help='which dataset to use [birds or flowers]') 
                        
    parser.add_argument('--load_from_epoch', type=int, default= 0, 
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default = None)
    parser.add_argument('--testing_path', type=str, default = None,
                        help='the h5 file that is used for evaluation')

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    
    vs_model    = ImgSenRanking(dim_image, sent_dim, hid_dim)
    img_encoder = ImageEncoder()

    device_id = getattr(args, 'device_id', 0)
    print('device_id: ', device_id)
    if args.cuda:
        vs_model    = vs_model.cuda(device_id)
        img_encoder = img_encoder.cuda(device_id)

        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
            
    weight_root = os.path.join('neudist/', args.model_name)

    test_nd(args.testing_path, weight_root, img_encoder, vs_model, args)

    
    
