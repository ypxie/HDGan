import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from ..proj_utils.network_utils import *
import math
import functools


class condEmbedding(nn.Module):
    def __init__(self, noise_dim, emb_dim):
        super(condEmbedding, self).__init__()
        
        self.noise_dim = noise_dim
        self.emb_dim = emb_dim
        self.linear  = nn.Linear(noise_dim, emb_dim*2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def sample_encoded_context(self, mean, logsigma, kl_loss=False):
    
        epsilon = Variable(torch.cuda.FloatTensor(mean.size()).normal_())
        stddev  = logsigma.exp()
        
        return epsilon.mul(stddev).add_(mean)

    def forward(self, inputs, kl_loss=True):
        '''
        inputs: (B, dim)
        return: mean (B, dim), logsigma (B, dim)
        '''
        out = self.relu(self.linear(inputs))
        mean = out[:, :self.emb_dim]
        log_sigma = out[:, self.emb_dim:]

        c = self.sample_encoded_context(mean, log_sigma)
        return c, mean, log_sigma

#-----------------------------------------------#
#    used to encode image into feature maps     #
#-----------------------------------------------#

class ImageDown(torch.nn.Module):

    def __init__(self, input_size, num_chan, out_dim):
        """
            Parameters:
            ----------
            input_size: int
                input image size, can be 64, or 128, or 256
            num_chan: int
                channel of input images.
            out_dim : int
                the dimension of generated image code.
        """

        super(ImageDown, self).__init__()
        self.__dict__.update(locals())

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        activ = nn.LeakyReLU(0.2, True)
        _layers = []

        # use large kernel_size at the end to prevent using zero-padding and stride
        if input_size == 64:
            cur_dim = 128
            _layers += [conv_norm(num_chan, cur_dim, norm_layer,
                                  stride=2, activation=activ, use_norm=False)]  # 32
            _layers += [conv_norm(cur_dim, cur_dim*2,
                                  norm_layer, stride=2, activation=activ)]  # 16
            _layers += [conv_norm(cur_dim*2, cur_dim*4,
                                  norm_layer, stride=2, activation=activ)]  # 8
            _layers += [conv_norm(cur_dim*4, out_dim,  norm_layer,
                                  stride=1, activation=activ, kernel_size=5, padding=0)]  # 4

        if input_size == 128:
            cur_dim = 64
            _layers += [conv_norm(num_chan, cur_dim, norm_layer,
                                  stride=2, activation=activ, use_norm=False)]  # 64
            _layers += [conv_norm(cur_dim, cur_dim*2,
                                  norm_layer, stride=2, activation=activ)]  # 32
            _layers += [conv_norm(cur_dim*2, cur_dim*4,
                                  norm_layer, stride=2, activation=activ)]  # 16
            _layers += [conv_norm(cur_dim*4, cur_dim*8,
                                  norm_layer, stride=2, activation=activ)]  # 8
            _layers += [conv_norm(cur_dim*8, out_dim,  norm_layer,
                                  stride=1, activation=activ, kernel_size=5, padding=0)]  # 4

        if input_size == 256:
            cur_dim = 32
            _layers += [conv_norm(num_chan, cur_dim, norm_layer,
                                  stride=2, activation=activ, use_norm=False)]  # 128
            _layers += [conv_norm(cur_dim, cur_dim*2,
                                  norm_layer, stride=2, activation=activ)]  # 64
            _layers += [conv_norm(cur_dim*2, cur_dim*4,
                                  norm_layer, stride=2, activation=activ)]  # 32
            _layers += [conv_norm(cur_dim*4, cur_dim*8,
                                  norm_layer, stride=2, activation=activ)]  # 16
            _layers += [conv_norm(cur_dim*8, cur_dim*8,
                                  norm_layer, stride=2, activation=activ)]  # 8

            _layers += [conv_norm(cur_dim*8, out_dim,  norm_layer,
                                  stride=1, activation=activ, kernel_size=5, padding=0)]  # 4

        self.node = nn.Sequential(*_layers)

    def forward(self, inputs):

        out = self.node(inputs)
        return out


class DiscClassifier(nn.Module):
    def __init__(self, enc_dim, emb_dim, kernel_size):
        """
            Parameters:
            ----------
            enc_dim: int
                the channel of image code.
            emb_dim: int
                the channel of sentence code.
            kernel_size : int
                kernel size used for final convolution.
        """

        super(DiscClassifier, self).__init__()
        self.__dict__.update(locals())
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        activ = nn.LeakyReLU(0.2, True)
        inp_dim = enc_dim + emb_dim

        _layers = [conv_norm(inp_dim, enc_dim, norm_layer, kernel_size=1, stride=1, activation=activ),
                   nn.Conv2d(enc_dim, 1, kernel_size=kernel_size, padding=0, bias=True)]

        self.node = nn.Sequential(*_layers)

    def forward(self, sent_code,  img_code):

        sent_code = sent_code.unsqueeze(-1).unsqueeze(-1)
        dst_shape = list(sent_code.size())
        dst_shape[1] = sent_code.size()[1]
        dst_shape[2] = img_code.size()[2]
        dst_shape[3] = img_code.size()[3]
        sent_code = sent_code.expand(dst_shape)
        comp_inp = torch.cat([img_code, sent_code], dim=1)
        output = self.node(comp_inp)
        chn = output.size()[1]
        output = output.view(-1, chn)

        return output


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        activ = nn.ReLU(True)

        seq = [pad_conv_norm(dim, dim, norm_layer, use_bias=use_bias, activation=activ),
               pad_conv_norm(dim, dim, norm_layer, use_activation=False, use_bias=use_bias)]
        self.res_block = nn.Sequential(*seq)

    def forward(self, input):
        return self.res_block(input) + input


class Sent2FeatMap(nn.Module):
    # used to project a sentence code into a set of feature maps
    def __init__(self, in_dim, row, col, channel, activ=None):
        super(Sent2FeatMap, self).__init__()
        self.__dict__.update(locals())
        out_dim = row*col*channel
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        _layers = [nn.Linear(in_dim, out_dim)]
        _layers += [norm_layer(out_dim)]
        if activ is not None:
            _layers += [activ]
        self.out = nn.Sequential(*_layers)

    def forward(self, inputs):
        output = self.out(inputs)
        output = output.view(-1, self.channel, self.row, self.col)
        return output


#----------------Define Generator and Discriminator-----------------------------#

class Generator(nn.Module):
    def __init__(self, sent_dim, noise_dim, emb_dim, hid_dim, num_resblock=1, side_output_at=[64, 128, 256]):
        """
        Parameters:
        ----------
        sent_dim: int
            the dimension of sentence embedding
        noise_dim: int
            the dimension of noise input
        emb_dim : int
            the dimension of compressed sentence embedding.
        hid_dim: int
            used to control the number of feature maps.
        num_resblock: int
            the scale factor of generator (see paper for explanation).
        side_output_at:  list
            contains local loss size for discriminator at scales.
        """

        super(Generator, self).__init__()
        self.__dict__.update(locals())

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        act_layer = nn.ReLU(True)
        self.condEmbedding = condEmbedding(sent_dim, emb_dim)

        self.vec_to_tensor = Sent2FeatMap(
            emb_dim+noise_dim, 4, 4, self.hid_dim*8)
        self.side_output_at = side_output_at

        # feature map dimension reduce at which resolution
        reduce_dim_at = [8, 32, 128, 256]
        # different scales for all networks
        num_scales = [4, 8, 16, 32, 64, 128, 256]

        cur_dim = self.hid_dim*8
        for i in range(len(num_scales)):
            seq = []
            # unsampling
            if i != 0:
                seq += [nn.Upsample(scale_factor=2, mode='nearest')]
            # if need to reduce dimension
            if num_scales[i] in reduce_dim_at:
                seq += [pad_conv_norm(cur_dim, cur_dim//2,
                                      norm_layer, activation=act_layer)]
                cur_dim = cur_dim//2
            # add residual blocks
            for n in range(num_resblock):
                seq += [ResnetBlock(cur_dim)]
            # add main convolutional module
            setattr(self, 'scale_%d' % (num_scales[i]), nn.Sequential(*seq))

            if num_scales[i] in self.side_output_at:
                setattr(self, 'tensor_to_img_%d' %
                        (num_scales[i]), branch_out(cur_dim))

        self.apply(weights_init)
        print('>> Init HDGAN Generator')
        print('\t side output at {}'.format(str(side_output_at)))

    def forward(self, sent_embeddings, z):
        """
        Parameters:
        ----------
        sent_embeddings: [B, sent_dim]
            sentence embedding obtained from char-rnn
        z: [B, noise_dim]
            noise input

        Returns:
        ----------
        out_dict: dictionary
            dictionary containing the generated images at scale [64, 128, 256]
        kl_loss: tensor
            Kullback–Leibler divergence loss from conditionining embedding
        """
        
        sent_random, mean, logsigma = self.condEmbedding(sent_embeddings) 

        text = torch.cat([sent_random, z], dim=1)

        x = self.vec_to_tensor(text)
        x_4 = self.scale_4(x)
        x_8 = self.scale_8(x_4)
        x_16 = self.scale_16(x_8)
        x_32 = self.scale_32(x_16)

        # skip 4x4 feature map to 32 and send to 64
        x_64 = self.scale_64(x_32)
        output_64 = self.tensor_to_img_64(x_64)

        # skip 8x8 feature map to 64 and send to 128
        x_128 = self.scale_128(x_64)
        output_128 = self.tensor_to_img_128(x_128)

        # skip 16x16 feature map to 128 and send to 256
        out_256 = self.scale_256(x_128)
        output_256 = self.tensor_to_img_256(out_256)

        return output_64, output_128, output_256, mean, logsigma


class Discriminator(torch.nn.Module):
    def __init__(self, num_chan,  hid_dim, sent_dim, emb_dim, side_output_at=[64, 128, 256]):
        """
        Parameters:
        ----------
        num_chan: int
            channel of generated images.
        enc_dim: int
            Reduce images inputs to (B, enc_dim, H, W) feature
        emb_dim : int
            the dimension of compressed sentence embedding.
        side_output_at:  list
            contains local loss size for discriminator at scales.
        """

        super(Discriminator, self).__init__()
        self.__dict__.update(locals())

        activ = nn.LeakyReLU(0.2, True)
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)

        self.side_output_at = side_output_at

        enc_dim = hid_dim * 4  # the ImageDown output dimension

        if 64 in side_output_at:  # discriminator for 64 input
            self.img_encoder_64 = ImageDown(64,  num_chan,  enc_dim)  # 4x4
            self.pair_disc_64 = DiscClassifier(enc_dim, emb_dim, kernel_size=4)
            self.local_img_disc_64 = nn.Conv2d(
                enc_dim, 1, kernel_size=4, padding=0, bias=True)
            _layers = [nn.Linear(sent_dim, emb_dim), activ]
            self.context_emb_pipe_64 = nn.Sequential(*_layers)

        if 128 in side_output_at:  # discriminator for 128 input
            self.img_encoder_128 = ImageDown(128,  num_chan, enc_dim)  # 4x4
            self.pair_disc_128 = DiscClassifier(
                enc_dim, emb_dim, kernel_size=4)
            # nn.Conv2d(enc_dim, 1, kernel_size=1, padding=0, bias=True)
            self.local_img_disc_128 = nn.Conv2d(
                enc_dim, 1, kernel_size=4, padding=0, bias=True)
            # map sentence to a code of length emb_dim
            _layers = [nn.Linear(sent_dim, emb_dim), activ]
            self.context_emb_pipe_128 = nn.Sequential(*_layers)

        if 256 in side_output_at:  # discriminator for 256 input
            self.img_encoder_256 = ImageDown(256, num_chan, enc_dim)     # 8x8
            self.pair_disc_256 = DiscClassifier(
                enc_dim, emb_dim, kernel_size=4)
            # shrink is used for mapping 8x8 Feature maps to 5x5
            # self.shrink_256 = conv_norm(enc_dim, enc_dim,  norm_layer, stride=1, activation=activ, kernel_size=5, padding=0)
            self.local_img_disc_256 = nn.Conv2d(
                enc_dim, 1, kernel_size=1, padding=0, bias=True)
            # map sentence to a code of length emb_dim
            _layers = [nn.Linear(sent_dim, emb_dim), activ]
            self.context_emb_pipe_256 = nn.Sequential(*_layers)

        self.apply(weights_init)

        print('>> Init HDGAN Discriminator')
        print('\t Apply adversarial loss at scale {}'.format(str(side_output_at)))

    def forward(self, images, embedding):
        '''
        Parameters:
        -----------
        images:    (B, C, H, W)
            input image tensor
        embedding : (B, sent_dim)
            corresponding embedding
        outptuts:  
        -----------
        out_dict: dict
            dictionary containing: pair discriminator output and image discriminator output
        '''
        out_dict = OrderedDict()
        this_img_size = images.size()[3]
        assert this_img_size in [
            32, 64, 128, 256], 'wrong input size {} in image discriminator'.format(this_img_size)

        img_encoder = getattr(self, 'img_encoder_{}'.format(this_img_size))
        local_img_disc = getattr(
            self, 'local_img_disc_{}'.format(this_img_size), None)
        pair_disc = getattr(self, 'pair_disc_{}'.format(this_img_size))
        context_emb_pipe = getattr(
            self, 'context_emb_pipe_{}'.format(this_img_size))

        sent_code = context_emb_pipe(embedding)
        img_code = img_encoder(images)

        pair_disc_out = pair_disc(sent_code, img_code)

        local_img_disc_out = local_img_disc(img_code)

        return pair_disc_out, local_img_disc_out
