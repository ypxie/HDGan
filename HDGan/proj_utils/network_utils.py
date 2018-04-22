import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque, OrderedDict
import functools
from .torch_utils import *

## Weights init function, DCGAN use 0.02 std
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: 
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)

def branch_out(in_dim, out_dim=3):
    _layers = [ nn.ReflectionPad2d(1),
                nn.Conv2d(in_dim, out_dim, 
                kernel_size = 3, padding=0, bias=False)]    
    _layers += [nn.Tanh()]

    return nn.Sequential(*_layers)
    
def KL_loss(mu, log_sigma):
    loss = -log_sigma + .5 * (-1 + torch.exp(2. * log_sigma) + mu**2)
    loss = torch.mean(loss)
    return loss

def sample_encoded_context(mean, logsigma, kl_loss=False, epsilon=None):
    """ 
    Sampling a vector from Norm(mean, sigma)
    Parameters:
    ----------
    mean: int
        mean vector.
    logsigma : int
        logsigma vector.
    kl_loss: bool
        whether to return kl_loss or not
    epsilon:
        a noise vector sampled from N(0, 1)
    """
    # epsilon = tf.random_normal(tf.shape(mean))
    if epsilon is None:
        epsilon = to_device( torch.randn(mean.size()), mean, requires_grad=False) 
    stddev  = torch.exp(logsigma)
    c = mean + stddev * epsilon

    kl_loss = KL_loss(mean, logsigma) if kl_loss else None
    return c, kl_loss
 
def pad_conv_norm(dim_in, dim_out, norm_layer, kernel_size=3, use_activation=True, 
                  use_bias=False, activation=nn.ReLU(True)):
    # designed for generators
    seq = []
    if kernel_size != 1:
        seq += [nn.ReflectionPad2d(1)]
        
    seq += [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=0, bias=use_bias),
            norm_layer(dim_out)]
    
    if use_activation:
        seq += [activation]
    
    return nn.Sequential(*seq)

def conv_norm(dim_in, dim_out, norm_layer, kernel_size=3, stride=1, use_activation=True, 
              use_bias=False, activation=nn.ReLU(True), use_norm=True,padding=None):
    # designed for discriminator
    
    if kernel_size == 3:
        padding = 1 if padding is None else padding
    else:
        padding = 0 if padding is None else padding

    seq = [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, bias=use_bias, stride=stride),
           ]
    if use_norm:
        seq += [norm_layer(dim_out)]
    if use_activation:
        seq += [activation]
    
    return nn.Sequential(*seq)
