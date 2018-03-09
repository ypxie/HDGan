import os
import sys, os
import numpy as np
sys.path.insert(0, os.path.join('..','..'))

home = os.path.expanduser('~')
proj_root = os.path.join('..','..')

data_root  = os.path.join(proj_root, 'Data', 'Results')
model_root = os.path.join(proj_root, 'Models')

import torch.multiprocessing as mp
from HDGan.proj_utils.local_utils import Indexflow
from HDGan.neuralDist.test_nd_worker import test_worker

# data_folder: the folder that contains the generated image (saved as h5 file)

if 1: #local [64, 256] 

    test_birds  =   \
                    { 'load_from_epoch': 110, 'batch_size': 8, 'device_id': 0,
                      "data_folder":os.path.join(data_root, 'birds','Finaltesting_num_1'), 
                      "result_marker": "Finaltesting_num_1", "dataset":"birds",
                      "file_name": "birds_256_G_epoch_500.h5",
                      'model_name':'neural_dist_birds',
                    }
    
    test_flowers  =   \
                    { 'load_from_epoch': 110, 'batch_size': 8, 'device_id': 0,
                      "data_folder":os.path.join(data_root, 'flowers','Finaltesting_num_1'), 
                      "result_marker": "Finaltesting_num_1", "dataset":"flowers",
                      "file_name": "flower_256_G_epoch_580.h5",
                      'model_name':'neural_dist_flowers',
                    }                  
                              
training_pool = np.array([
                    test_birds,
                    test_flowers,
                 ])

show_progress = 0
processes = []
Totalnum = len(training_pool)

for select_ind in Indexflow(Totalnum, 2, random=False):
    select_pool = training_pool[select_ind]

    for this_dick in select_pool:
        p = mp.Process(target=test_worker, args= (data_root, model_root, this_dick) )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Finish the round with: ', select_ind)

