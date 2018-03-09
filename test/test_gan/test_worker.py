# -*- coding: utf-8 -*-

import numpy as np
import argparse, os
import torch, h5py

import torch.nn as nn
from collections import OrderedDict
from .proj_utils.local_utils import mkdirs
from .testGan import test_gans
from .fuel.datasets import TextDataset
from HDGan.models.hd_networks import Generator

def test_worker(data_root, model_root, save_root, testing_dict):
    print('testing_dict: ', testing_dict)
    
    parser = argparse.ArgumentParser(description = 'Gans')    

    parser.add_argument('--batch_size', type=int, default=testing_dict['batch_size'], metavar='N',
                        help='batch size.')
    parser.add_argument('--device_id', type=int, default= testing_dict['device_id'], 
                        help='which device')
    parser.add_argument('--load_from_epoch', type=int, default= testing_dict['load_from_epoch'], 
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default = testing_dict['model_name'])

    parser.add_argument('--test_sample_num', type=int, default= testing_dict['test_sample_num'], 
                        help='The number of runs for each embeddings when testing')
                 
    parser.add_argument('--save_spec', type=str, default=testing_dict['save_spec'], help='save_spec')
    
    
    args = parser.parse_args()
    
    args.cuda = torch.cuda.is_available()

    netG = Generator(sent_dim=1024, noise_dim= 100, emb_dim=128, hid_dim=128, num_resblock=2)      
    
    data_name  = testing_dict['dataset']
    datadir    = os.path.join(data_root, data_name)

    device_id = getattr(args, 'device_id', 0)
    
    if args.cuda:
        netG = netG.cuda(device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    print ('>> initialize dataset')
    dataset = TextDataset(datadir, 'cnn-rnn', 4)
    filename_test = os.path.join(datadir, 'test')
    dataset.test = dataset.get_data(filename_test)

    model_name = args.model_name  
    
    save_folder  = os.path.join(save_root, data_name, args.save_spec + 'testing_num_{}'.format(args.test_sample_num) )
    mkdirs(save_folder)
    
    test_gans(dataset, model_root, model_name, save_folder, netG, args)

    
    