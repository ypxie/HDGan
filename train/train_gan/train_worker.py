# -*- coding: utf-8 -*-
import numpy as np
import argparse, os, sys
import torch, h5py
sys.path.insert(0, os.path.join('..','..'))

home = os.path.expanduser("~")
proj_root = os.path.join('..','..')
data_root  = os.path.join(proj_root, 'Data')
model_root = os.path.join(proj_root, 'Models')

import torch.nn as nn
from collections import OrderedDict

from HDGan.models.hd_networks import Generator
from HDGan.models.hd_networks import Discriminator 

from HDGan.HDGan import train_gans
from HDGan.fuel.datasets import TextDataset

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Gans')    

    parser.add_argument('--reuse_weights',    action='store_true',  default= False, help='continue from last checkout point')
    parser.add_argument('--load_from_epoch', type=int, default= 0,  help='load from epoch')

    parser.add_argument('--batch_size', type=int,      default= 16, metavar='N', help='batch size.')
    parser.add_argument('--device_id',  type=int,      default= 0,  help='which device')
    
    parser.add_argument('--model_name', type=str,      default= None)
    parser.add_argument('--dataset',    type=str,      default= None, help='which dataset to use [birds or flowers]') 
    
    parser.add_argument('--num_resblock', type=int, default = 1, help='number of resblock')
    parser.add_argument('--epoch_decay', type=float, default=100, help='decay learning rate by half every epoch_decay')

    
    parser.add_argument('--maxepoch', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--g_lr', type=float, default = 0.0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--d_lr', type=float, default = 0.0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--save_freq', type=int, default= 3, metavar='N',
                        help='how frequent to save the model')
    parser.add_argument('--display_freq', type=int, default= 200, metavar='N',
                        help='plot the results every {} batches')
    parser.add_argument('--verbose_per_iter', type=int, default= 50, 
                        help='print losses per iteration')
    parser.add_argument('--num_emb', type=int, default=4, metavar='N',
                        help='number of emb chosen for each image during training.')
    
    parser.add_argument('--noise_dim', type=int, default= 100, metavar='N',
                        help='the dimension of noise.')

    parser.add_argument('--ncritic', type=int, default= 1, metavar='N',
                        help='the channel of each image.')
    parser.add_argument('--test_sample_num', type=int, default=4, 
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--KL_COE', type=float, default= 4, metavar='N',
                        help='kl divergency coefficient.')
    ## add more
    args = parser.parse_args()
    args.cuda  = torch.cuda.is_available()
    
    data_name  = args.dataset
    datadir = os.path.join(data_root, data_name)

    # Generator
    netG = Generator(sent_dim=1024, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, num_resblock = 1)
    
    # Discriminator
    netD = Discriminator(num_chan = 3, hid_dim = 128, sent_dim=1024, emb_dim=128)
    
    print(args)

    device_id = getattr(args, 'device_id', 0)

    if args.cuda:
        netD = netD.cuda(device_id)
        netG = netG.cuda(device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        
    print ('>> initialize dataset')
    dataset = TextDataset(datadir, 'cnn-rnn', 4)
    filename_test = os.path.join(datadir, 'test')
    dataset.test = dataset.get_data(filename_test)
    filename_train = os.path.join(datadir, 'train')
    dataset.train = dataset.get_data(filename_train)
    
    model_name ='{}_{}_256'.format(args.model_name, data_name)
    print ('>> START training ')
    train_gans(dataset, model_root, model_name, netG, netD, args)
    