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

from torch.multiprocessing import Pool

import scipy
import time, json
import random 

TINY = 1e-8



def train_nd(dataset, model_root, mode_name, img_encoder, vs_model, args):
    lr = args.lr
    tot_epoch = args.maxepoch
    train_sampler = dataset.train.next_batch
    test_sampler  = dataset.test.next_batch

    train_num = dataset.train._num_examples
    test_num  = dataset.test._num_examples 

    number_example = train_num +  test_num
    prob_use_train = float(train_num)/number_example

    pool = Pool(3)
    trans_func = get_trans(img_encoder)
    updates_per_epoch = int(number_example / args.batch_size)
        
    ''' configure optimizer '''
    optimizer = optim.Adam(vs_model.parameters(), lr= args.lr, betas=(0.5, 0.999) )
    model_folder = os.path.join(model_root, mode_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    ''' load model '''
    if args.reuse_weights :
        weightspath = os.path.join(model_folder, 'W_epoch{}.pth'.format(args.load_from_epoch))
        
        if os.path.exists(weightspath) :
            weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
            print('reload weights from {}'.format(weightspath))
            vs_model.load_state_dict(weights_dict)
            start_epoch = args.load_from_epoch + 1
        else:
            print ('{} do not exist!!'.format(weightspath))
            raise NotImplementedError
    else:
        start_epoch = 1

    loss_plot = plot_scalar(name = "loss", env= mode_name, rate = args.display_freq)
    lr_plot   = plot_scalar(name = "lr", env= mode_name, rate = args.display_freq)

    for epoch in range(start_epoch, tot_epoch):
        start_timer = time.time()
        # learning rate
        if epoch % args.epoch_decay == 0:
            lr = lr/2
            set_lr(optimizer, lr)

        for it in range(updates_per_epoch):
            vs_model.train()
            img_encoder.eval()

            if np.random.random() >= prob_use_train:
                images, wrong_images, np_embeddings, _, _ = train_sampler(args.batch_size, args.num_emb)
            else:
                images, wrong_images, np_embeddings, _, _ = test_sampler(args.batch_size, args.num_emb)
            
            img_224     =  pre_process(images["output_256"], pool, trans_func)
            
            embeddings  = to_device(np_embeddings, vs_model.device_id, requires_grad=False)
            img_224     = to_device(img_224, img_encoder.device_id, volatile=True)
            
            img_feat   = img_encoder(img_224)
            
            img_feat   = img_feat.squeeze(-1).squeeze(-1)

            
            img_feat   = to_device(img_feat.data,vs_model.device_id, requires_grad=True)
            sent_emb, img_emb = vs_model(embeddings, img_feat)
            cost = PairwiseRankingLoss(img_emb, sent_emb, args.margin)

            optimizer.zero_grad()
            cost.backward()
            
            optimizer.step()

            cost_val = cost.cpu().data.numpy().mean()

            loss_plot.plot(cost_val)
            lr_plot.plot(lr)
            end_timer = time.time() - start_timer
            if it % args.verbose_per_iter == 0:
                print ('[epoch %d/%d iter %d]: lr = %.6f cost_val = %.5f' % (epoch, tot_epoch, it, lr, cost_val))
                sys.stdout.flush()

        if epoch % args.save_freq == 0:
            vs_model = vs_model.cpu()
            torch.save(vs_model.state_dict(), os.path.join(model_folder, 'W_epoch{}.pth'.format(epoch)))
            print('save weights at {}'.format(model_folder))
            vs_model = vs_model.cuda(args.device_id)
        print ('epoch {}/{} finished [time = {}s] ...'.format(epoch, tot_epoch, end_timer))

def PairwiseRankingLoss(im, sent, margin):
    # compute image-sentence score matrix
    scores = torch.mm(im, sent.transpose(1, 0))
    diagonal = scores.diag()
    batch_size = scores.size()[0]

    sent_zeros = Variable(sent.data.new(scores.size()[0], scores.size()[1]).fill_(0.0)  )
    img_zeros  = Variable(im.data.new(scores.size()[0], scores.size()[1]).fill_(0.0)  )
    
    cost_s = torch.max(sent_zeros, (margin-diagonal).expand_as(scores)+scores)
    cost_im = torch.max(img_zeros, (margin-diagonal).expand_as(scores).transpose(1, 0)+scores)

    for i in range(scores.size()[0]):
        cost_s[i, i] = 0
        cost_im[i, i] = 0

    return (cost_s.sum() + cost_im.sum())

def resize_images(img, dst_shape):
    tmp = scipy.misc.imresize(img, dst_shape)
    return tmp

def get_trans(img_encoder):
    img_tensor_list = []
    trans = transforms.Compose([
                transforms.ToTensor(),
                #transforms.RandomSizedCrop(224),
                transforms.Normalize(mean=img_encoder.mean, std=img_encoder.std),
                ])
    return trans

def _process(inputs):
    this_img, trans =inputs
    this_crop = resize_images(this_img, (299, 299))
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
