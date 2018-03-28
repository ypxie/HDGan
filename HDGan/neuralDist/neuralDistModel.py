# coding: utf-8
import numpy as np
import torch
from torch.autograd import Variable

import torch.nn as nn
from .pretrainedmodels import inceptionresnetv2 

def l2norm(input, p=2.0, dim=1, eps=1e-12):
    """
    Compute L2 norm, row-wise
    """
    #print("input size(): ", input.size())
    l2_inp = input / input.norm(p, dim, keepdim=True).clamp(min=eps)
    return l2_inp.expand_as(input)

def xavier_weight(tensor):
    if isinstance(tensor, Variable):
        xavier_weight(tensor.data)
        return tensor

    nin, nout = tensor.size()[0], tensor.size()[1]
    r = np.sqrt(6.) / np.sqrt(nin + nout)
    return tensor.normal_(0, r)


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.register_buffer('device_id', torch.IntTensor(1))
        resmodel = inceptionresnetv2(1000)

        self.encoder = nn.Sequential(
                    *list(resmodel.children())[:-1]
                )
        self.mean = resmodel.mean
        self.std = resmodel.std
        self.input_size =  resmodel.input_size

    def forward(self, x):
        feat = self.encoder(x)
        return feat

class ImgSenRanking(torch.nn.Module):
    def __init__(self, dim_image, sent_dim,  hid_dim):
        super(ImgSenRanking, self).__init__()
        self.register_buffer('device_id', torch.IntTensor(1))

        self.linear_img = torch.nn.Linear(dim_image, hid_dim)
        self.linear_sent = torch.nn.Linear(sent_dim, hid_dim)

        self.init_weights()

    def init_weights(self):
        xavier_weight(self.linear_img.weight)
        xavier_weight(self.linear_sent.weight)
        self.linear_img.bias.data.fill_(0)
        self.linear_sent.bias.data.fill_(0)

    def forward(self, sent, img):
        
        img_vec   = self.linear_img(img)    
        sent_vec = self.linear_sent(sent)   
        return l2norm(sent_vec), l2norm(img_vec)
        

