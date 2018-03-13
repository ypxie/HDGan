# -*- coding: utf-8 -*-
import numpy as np
import argparse, os
import torch, h5py
sys.path.insert(0, os.path.join('..','..'))

home = os.path.expanduser("~")
proj_root = os.path.join('..','..')

data_root  = os.path.join(proj_root, 'Data')
model_root = os.path.join(proj_root, 'Models')


import torch.nn as nn
from collections import OrderedDict

from HDGan.neuralDist.trainNeuralDist  import train_nd
from HDGan.neuralDist.neuralDistModel  import ImgSenRanking
from HDGan.neuralDist.neuralDistModel  import ImageEncoder
from HDGan.fuel.datasets import TextDataset

    
if  __name__ == '__main__':
    dim_image   =  1536
    sent_dim    =  1024
    hid_dim     =  512
    
    parser = argparse.ArgumentParser(description = 'NeuralDist')    
    parser.add_argument('--weight_decay', type=float, default= 0,
                        help='weight decay for training')
    parser.add_argument('--maxepoch', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default = .0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    
    parser.add_argument('--save_freq', type=int, default= 5, metavar='N',
                        help='how frequent to save the model')
    parser.add_argument('--display_freq', type=int, default= 200, metavar='N',
                        help='plot the results every {} batches')
    parser.add_argument('--verbose_per_iter', type=int, default= 50, 
                        help='print losses per iteration')
    
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size.')
    parser.add_argument('--num_emb', type=int, default=1, metavar='N',
                        help='number of emb chosen for each image.')
    ## add more
    parser.add_argument('--device_id', type=int, default=0, 
                        help='which device')
    
    parser.add_argument('--reuse_weights',    action='store_true',  default= False, 
                        help='continue from last checkout point')
    parser.add_argument('--load_from_epoch', type=int, default= 0, 
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, 
                        default= 'neural_dist')
    
    parser.add_argument('--dataset', type=str, default=None, 
                        help='which dataset to use [birds or flowers]') 
    parser.add_argument('--margin',  default = 0.2, 
                        help='which devices to parallel the data')

    args = parser.parse_args()

    args.cuda  = torch.cuda.is_available()
    
    data_name  = args.dataset
    datadir = os.path.join(data_root, data_name)

    vs_model    = ImgSenRanking(dim_image, sent_dim, hid_dim)
    img_encoder = ImageEncoder()

    device_id = getattr(args, 'device_id', 0)

    if args.cuda:
        vs_model    = vs_model.cuda(device_id)
        img_encoder = img_encoder.cuda(device_id)

        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    
    print ('>> initialize dataset')
    dataset = TextDataset(datadir, 'cnn-rnn', 4)
    filename_test = os.path.join(datadir, 'test')
    dataset.test = dataset.get_data(filename_test)
    filename_train = os.path.join(datadir, 'train')
    dataset.train = dataset.get_data(filename_train)
  
    #model_name ='{}_{}_{}'.format(args.model_name, data_name, args.imsize)
    model_name ='{}_{}'.format(args.model_name, data_name)
    print ('>> START training ')
    train_nd(dataset, model_root, model_name, img_encoder, vs_model, args)
