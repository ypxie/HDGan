import numpy as np
import os, sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter
import torchvision.transforms as transforms

from torch.nn.utils import clip_grad_norm
from ..proj_utils.plot_utils import *
from ..proj_utils.torch_utils import *
from ..proj_utils.local_utils import Indexflow, IndexH5

from torch.multiprocessing import Pool

import scipy
import time, json
import random, h5py

TINY = 1e-8


def get_trans(img_encoder):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    img_tensor_list = []
    
    trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=img_encoder.mean, std=img_encoder.std),
                ])
    return trans

def _process(inputs):
    this_img, trans =inputs
    this_crop = scipy.misc.imresize(this_img, (299, 299))
    this_img_tensor = trans(this_crop)
    return this_img_tensor

def pre_process(images, pool, trans=None):
    
    images = (images + 1) /2 * 255
    images = images.transpose(0, 2,3,1)
    bs = images.shape[0]
    img_tensor_list = []
    targets = pool.imap(_process,  ( (images[idx], trans) for idx in range(bs) ) )
    
    for idx in range(bs):
        this_img_tensor = targets.__next__()
        img_tensor_list.append(this_img_tensor)

    img_tensor_all = torch.stack(img_tensor_list,0)
    return img_tensor_all


def test_nd(h5_path, model_root, mode_name, img_encoder, vs_model, args):
    h5_folder, h5_name = os.path.split(h5_path)
    h5_name_noext = os.path.splitext(h5_name)[0]
    result_path  = os.path.join(h5_folder, h5_name_noext+"_epoch_{}_neu_dist.json".format(args.load_from_epoch))
    ranking_path = os.path.join(h5_folder, h5_name_noext+"_epoch_{}_nd_ranking.json".format(args.load_from_epoch))
    print("{} exists or not: ".format(h5_path), os.path.exists(h5_path))
    with h5py.File(h5_path,'r') as h5_data:
        pool = Pool(3)
        all_embeddings = h5_data["embedding"]
        images = h5_data["output_256"]
        class_IDs = h5_data["classIDs"]
        try:
            saveIDs = h5_data["saveIDs"]
        except:
            print('you dont have saveIDs. now we use fake ones')
            saveIDs = class_IDs

        name_list = ["{}_{}".format(this_cls, this_save) for (this_cls, this_save) in zip(class_IDs, saveIDs)]

        all_keys = []
        for this_key in h5_data.keys():
            if "output" in this_key:
                all_keys.append(this_key)
        
        trans_func = get_trans(img_encoder)
        model_folder = os.path.join(model_root, mode_name)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        ''' load model '''
        weightspath = os.path.join(model_folder, 'W_epoch{}.pth'.format(args.load_from_epoch))
        
        if os.path.exists(weightspath) :
            weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
            print('reload weights from {}'.format(weightspath))
            vs_model.load_state_dict(weights_dict)# 12)
            start_epoch = args.load_from_epoch + 1
        else:
            print ('{} do not exist!!'.format(weightspath))
            raise NotImplementedError

        vs_model.eval()
        img_encoder.eval()
        all_results = {}
        for this_key in all_keys:
            this_images = h5_data[this_key] 
            num_imgs = this_images.shape[0]
            all_distance = 0
            all_cost_list = []
            print("Now processing {}".format(this_key))
            for thisInd in Indexflow(num_imgs, args.batch_size, random=False): #
                this_batch    = IndexH5(this_images, thisInd) 
                np_embeddings = IndexH5(all_embeddings, thisInd)

                img_299     =  pre_process(this_batch, pool, trans_func)
                
                embeddings  =  to_device(np_embeddings, vs_model.device_id, volatile=True)
                img_299     =  to_device(img_299, img_encoder.device_id, volatile=True)
               
                img_feat    =  img_encoder(img_299)
                
                img_feat    =  img_feat.squeeze(-1).squeeze(-1)

                img_feat   = to_device(img_feat.data,vs_model.device_id, volatile=True)

                sent_emb, img_emb = vs_model(embeddings, img_feat)
                
                cost     = torch.sum(img_emb*sent_emb, 1, keepdim=False)
                cost_val = cost.cpu().data.numpy()
                all_cost_list.append(cost_val)

            all_cost = np.concatenate(all_cost_list, 0)    
            cost_mean = float(np.mean(all_cost))
            cost_std  = float(np.std(all_cost))

            all_results[this_key] = {"mean":cost_mean, "std":cost_std}
        
        rank_idx = np.argsort(-all_cost) #from large to low
        
        sorted_name = [name_list[idx] for idx in rank_idx]

        print(all_results)    
        with open(result_path, 'w') as f:
            json.dump(all_results, f)
        with open(ranking_path, 'w') as f:
            json.dump(sorted_name, f)