# -*- coding: utf-8 -*-

import numpy as np
import argparse, os
import torch, h5py

import torch.nn as nn
from collections import OrderedDict
from .proj_utils.local_utils import mkdirs
from .testGan import test_gans
from .fuel.datasets import TextDataset

def test_worker(data_root, model_root, save_root, testing_dict):
    print('testing_dict: ', testing_dict)
    
    reduce_dim_at       = testing_dict.get('reduce_dim_at')
    batch_size          = testing_dict.get('batch_size')
    device_id           = testing_dict.get('device_id')
    test_sample_num     = testing_dict.get('test_sample_num', 10)
    save_images         = testing_dict.get('save_images', False)
    imsize              = testing_dict.get('imsize')
    num_resblock        = testing_dict.get('num_resblock', 2)
    
    data_root           = testing_dict.get('data_root', data_root)
    model_root          = testing_dict.get('model_root', model_root)     

    parser = argparse.ArgumentParser(description = 'Gans')    

    parser.add_argument('--noise_dim', type=int, default= 100, metavar='N',
                        help='dimension of gaussian noise.')

    parser.add_argument('--batch_size', type=int, default=batch_size, metavar='N',
                        help='batch size.')
    parser.add_argument('--num_emb', type=int, default=4, metavar='N',
                        help='number of emb chosen for each image.')
    ## add more
    parser.add_argument('--device_id', type=int, default= device_id, 
                        help='which device')
    parser.add_argument('--imsize', type=int, default=imsize, 
                        help='output image size')
    parser.add_argument('--epoch_decay', type=float, default=100, 
                        help='decay epoch image size')
    parser.add_argument('--load_from_epoch', type=int, default= testing_dict['load_from_epoch'], 
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default = testing_dict['model_name'])

    parser.add_argument('--test_sample_num', type=int, default= test_sample_num, 
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--norm_type', type=str, default='bn', 
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--gen_activation_type', type=str, default='relu', 
                        help='The number of runs for each embeddings when testing')
                        
    parser.add_argument('--which_gen', type=str, default=testing_dict['which_gen'],  help='generator type')
    parser.add_argument('--which_disc', type=str, default=testing_dict['which_disc'], help='discriminator type')
    parser.add_argument('--save_spec', type=str, default=testing_dict['save_spec'], help='save_spec')
    parser.add_argument('--train_mode', type=bool, default = testing_dict['train_mode'],
                        help='continue from last checkout point')
    parser.add_argument('--save_images', type=bool, default = save_images,
                        help='do you really want to save big images to folder')
    
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    
    # Generator
    if args.which_gen == 'origin':
        from HDGan.models.hd_networks import Generator
        netG = Generator(sent_dim=1024, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, 
                         norm=args.norm_type, activation=args.gen_activation_type, 
                         output_size=args.imsize, reduce_dim_at=reduce_dim_at)      
    else:
        raise NotImplementedError('Generator [%s] is not implemented' % args.which_gen)
        
    data_name  = testing_dict['dataset']
    datadir    = os.path.join(data_root, data_name)

    device_id = getattr(args, 'device_id', 0)
    print('device_id: ', device_id)
    if args.cuda:
        #netD = netD.cuda(device_id)
        netG = netG.cuda(device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    print ('>> initialize dataset')
    dataset = TextDataset(datadir, 'cnn-rnn', 4)
    filename_test = os.path.join(datadir, 'test')
    dataset.test = dataset.get_data(filename_test)

    model_name = args.model_name   #'{}_{}_{}'.format(args.model_name, data_name, args.imsize)
    
    save_folder  = os.path.join(save_root, data_name, args.save_spec + 'testing_num_{}'.format(args.test_sample_num) )
    mkdirs(save_folder)
    
    test_gans(dataset, model_root, model_name, save_folder, netG, args)

    
    