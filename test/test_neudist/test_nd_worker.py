# -*- coding: utf-8 -*-

import numpy as np
import argparse, os
import torch, h5py

import torch.nn as nn
from collections import OrderedDict
from ..proj_utils.local_utils import mkdirs
from .testNeuralDist  import test_nd
from .neuralDistModel  import ImgSenRanking
from .neuralDistModel  import ImageEncoder

def test_worker(data_root, model_root, testing_dict):
    print('testing_dict: ', testing_dict)

    batch_size   =  testing_dict.get('batch_size')
    device_id    =  testing_dict.get('device_id')
    
    dim_image    =  testing_dict.get('dim_image', 1536) 
    sent_dim     =  testing_dict.get('sent_dim',  1024) 
    hid_dim      =  testing_dict.get('hid_dim',    512) 
        
    data_root           = testing_dict.get('data_root', data_root)
    model_root          = testing_dict.get('model_root', model_root)    
    
    parser = argparse.ArgumentParser(description = 'test nd') 
    parser.add_argument('--batch_size', type=int, default=batch_size, metavar='N',
                        help='batch size.')
    
    parser.add_argument('--device_id', type=int, default= device_id, 
                        help='which device')
    
    parser.add_argument('--load_from_epoch', type=int, default= testing_dict['load_from_epoch'], 
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default = testing_dict['model_name'])
    
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
            
    model_name = args.model_name   #'{}_{}_{}'.format(args.model_name, data_name, args.imsize)
    
    testing_path = os.path.join(testing_dict['data_folder'],  testing_dict['file_name'])

    test_nd(testing_path, model_root, model_name, img_encoder, vs_model, args)

    
    
