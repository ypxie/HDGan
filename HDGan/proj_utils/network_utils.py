import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from .torch_utils import *
# from .local_utils import Indexflow, split_img, imshow
from collections import deque, OrderedDict
import functools


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

def getNormLayer(norm='bn', dim=2):

    norm_layer = None
    if dim == 2:
        if norm == 'bn':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm == 'ln':
            norm_layer = functools.partial(LayerNorm2d)
    elif dim == 1:
        if norm == 'bn':
            norm_layer = functools.partial(nn.BatchNorm1d, affine=True)
        elif norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm1d, affine=False)
        elif norm == 'ln':
            norm_layer = functools.partial(LayerNorm1d)
    assert(norm_layer != None)
    return norm_layer

def to_variable(x, requires_grad=True,  var=True,volatile=False):
    
    if type(x) is Variable:
        return x
    if type(x) is np.ndarray:
        x = torch.from_numpy(x.astype(np.float32))
    if var:
        x = Variable(x, requires_grad=requires_grad, volatile=volatile)
    x.volatile = volatile 
    
    return x

def to_device(src, ref, var = True, volatile = False, requires_grad=True):
    requires_grad = requires_grad and (not volatile)
    src = to_variable(src, var=var, volatile=volatile,requires_grad=requires_grad)
    return src.cuda(ref.get_device()) if ref.is_cuda else src

def branch_out2(in_dim, out_dim=3):
    _layers = [nn.ReflectionPad2d(1),
                nn.Conv2d(in_dim, out_dim, 
                kernel_size = 3, padding=0, bias=False)]    
    _layers += [nn.Tanh()]

    return nn.Sequential(*_layers)
    
def KL_loss(mu, log_sigma):
    loss = -log_sigma + .5 * (-1 + torch.exp(2. * log_sigma) + mu**2)
    loss = torch.mean(loss)
    # kld = mu.pow(2).add_(log_sigma.exp()).mul_(-1).add_(1).add_(log_sigma)
    # loss = torch.mean(kld).mul_(-0.5)

    return loss

def sample_encoded_context(mean, logsigma, kl_loss=False, epsilon=None):
    
    # epsilon = tf.random_normal(tf.shape(mean))
    if epsilon is None:
        epsilon = to_device( torch.randn(mean.size()), mean, requires_grad=False) 
    stddev  = torch.exp(logsigma)
    c = mean + stddev * epsilon

    kl_loss = KL_loss(mean, logsigma) if kl_loss else None
    return c, kl_loss

class condEmbedding(nn.Module):
    def __init__(self, noise_dim, emb_dim, use_cond=True):
        super(condEmbedding, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.noise_dim = noise_dim
        self.emb_dim = emb_dim
        self.use_cond = use_cond
        if use_cond:
            self.linear  = nn.Linear(noise_dim, emb_dim*2)
        else:
            self.linear  = nn.Linear(noise_dim, emb_dim)
    def forward(self, inputs, kl_loss=True, epsilon=None):
        '''
        inputs: (B, dim)
        return: mean (B, dim), logsigma (B, dim)
        '''
        #print('cont embedding',inputs.get_device(),  self.linear.weight.get_device())
        out = F.leaky_relu( self.linear(inputs), 0.2, inplace=True )
        
        if self.use_cond:
            mean = out[:, :self.emb_dim]
            log_sigma = out[:, self.emb_dim:]

            c, kl_loss = sample_encoded_context(mean, log_sigma, kl_loss, epsilon)
            return c, kl_loss
        else:
            return out, 0

def genAct():
    return nn.ReLU(True)
def discAct():
    return nn.LeakyReLU(0.2, True)
def get_activation_layer(name):
    if name == 'lrelu':
        act_layer = nn.LeakyReLU(0.2, inplace=True)
    else:
        act_layer = nn.ReLU(True)
    return act_layer
    
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
