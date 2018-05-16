# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import sys
import torch
import h5py
sys.path.insert(0, os.path.join('..', '..'))

proj_root = os.path.join('..', '..')
data_root = os.path.join(proj_root, 'Data')
model_root = os.path.join(proj_root, 'Models')

import torch.nn as nn
from collections import OrderedDict

from HDGan.models.hd_networks import Generator
from HDGan.models.hd_networks import Discriminator

from HDGan.HDGan import train_gans
from HDGan.fuel.datasets import Dataset


datadir = os.path.join(data_root, 'coco')
finest_size = 256
num_emb = 4
batch_size = 128

dataset_train = Dataset(datadir, img_size=finest_size,
                        batch_size=batch_size, n_embed=num_emb, mode='train')
dataset_test = Dataset(datadir, img_size=finest_size,
                        batch_size=batch_size, n_embed=1, mode='test')

train_sampler = iter(dataset_test)

for i in range(9999):
    print(i)
    try:
        images, wrong_images, np_embeddings, _, filenames = next(train_sampler)
    # print(filenames)
    except:
        train_sampler = iter(dataset_test)
