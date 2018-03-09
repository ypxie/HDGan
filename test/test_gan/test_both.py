import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..','..'))

home = os.path.expanduser('~')
proj_root = os.path.join('..','..')
save_root  =  os.path.join(proj_root, 'Data', 'Results')

import torch.multiprocessing as mp
from HDGan.proj_utils.local_utils import Indexflow
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
                 
training_pool = np.array([
                  final_model_original_birds,
                  final_model_original_flowers
                 ])

show_progress = 0
processes = []
Totalnum = len(training_pool)

for select_ind in Indexflow(Totalnum, 4, random=False):
    select_pool = training_pool[select_ind]

    for this_dick in select_pool:

        p = mp.Process(target=test_worker, args= (data_root, model_root, save_root, this_dick) )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finish the round with: ', select_ind)

