import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..','..'))

home = os.path.expanduser('~')
proj_root = os.path.join('..','..')
save_root  =  os.path.join(proj_root, 'Data', 'Results')

from HDGan.test_worker import test_worker

if 1: # FINAL Model
    save_spec = 'Final'
    data_root = os.path.join(proj_root, 'Data')
    model_root = os.path.join(proj_root, 'Models')
    #10;
    final_model_original_birds  =   \
                   { 'test_sample_num' : 10,  'load_from_epoch': 500, 'dataset':'birds', "save_images":True, 
                     'device_id': 0, 'imsize':[64,128, 256], 'model_name':'birds_256',
                     'train_mode': False,  'save_spec': save_spec, 'batch_size': 2, 'which_gen': 'origin',
                     'which_disc':None, 'reduce_dim_at':[8, 32, 128, 256] }
    #26                 
    final_model_original_flowers  =   \
                   {'test_sample_num' : 26,  'load_from_epoch': 580, 'dataset':'flowers', "save_images":True, 
                    'device_id': 0,'imsize':[64,128, 256], 'model_name':'flower_256',
                    'train_mode': False,  'save_spec': save_spec, 'batch_size': 2, 'which_gen': 'origin',
                     'which_disc':'origin', 'reduce_dim_at':[8, 32, 128, 256] }

test_worker(data_root, model_root, save_root, final_model_original_birds)

test_worker(data_root, model_root, save_root, final_model_original_flowers)
