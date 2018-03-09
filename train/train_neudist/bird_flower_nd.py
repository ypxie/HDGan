import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..','..'))

home = os.path.expanduser("~")
proj_root = os.path.join('..','..')

data_root  = os.path.join(proj_root, 'Data')
model_root = os.path.join(proj_root, 'Models')

from HDGan.neuralDist.train_nd_worker import train_worker


bird_neudist  = { 'reuse_weights': False, 'batch_size': 64, 'device_id': 0, 'lr': .0002/(2**0),
                   'load_from_epoch': 0, 'model_name':'neural_dist', 
                  'dataset':'birds',
                }

flower_neudist  = { 'reuse_weights': False, 'batch_size': 64, 'device_id': 0, 'lr': .0002/(2**0),
                    'load_from_epoch': 0, 'model_name':'neural_dist', 
                    'dataset':'flowers',
                  }

train_worker(data_root, model_root, bird_neudist)

train_worker(data_root, model_root, flower_neudist)