import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from ..proj_utils.network_utils import *
import math

class Generator(nn.Module):
    def __init__(self, sent_dim, noise_dim, emb_dim, hid_dim, norm='bn', activation='relu', 
                 output_size=256,  reduce_dim_at= [8, 32, 128, 256], num_resblock = 1,
                 detach_list=[], use_cond = True):

        super(Generator, self).__init__()
        print('locals of gen: ', locals())
        self.__dict__.update(locals())
        norm_layer = getNormLayer(norm)
        act_layer = get_activation_layer(activation)

        self.register_buffer('device_id', torch.IntTensor(1))
        self.condEmbedding = condEmbedding(sent_dim, emb_dim, use_cond=use_cond)
        self.vec_to_tensor = Sent2FeatMap(emb_dim+noise_dim, 4, 4, self.hid_dim*8, norm=norm)
        #self.use_upsamle_skip = use_upsamle_skip

        if type(output_size) is int:
            if output_size==256: self.side_output_at = [64, 128, 256] 
            if output_size==128: self.side_output_at = [64, 128] 
            if output_size==64:  self.side_output_at = [64] 
        else:
            self.side_output_at = output_size
        # 64, 128, or 256 version
        self.max_output_size = max(self.side_output_at)

        if self.max_output_size == 256:
            num_scales = [4, 8, 16, 32, 64, 128, 256]
            text_upsampling_at = [4, 8, 16] 
        elif self.max_output_size == 128:
            num_scales = [4, 8, 16, 32, 64, 128]
            text_upsampling_at = [4, 8] 
        elif self.max_output_size == 64:
            num_scales = [4, 8, 16, 32, 64]
            text_upsampling_at = [4]     
            
        #reduce_dim_at  = [8, 32, 128, 256] # [8, 64, 256]
        #self.modules = OrderedDict()
        #self.side_modules = OrderedDict()

        cur_dim = self.hid_dim*8
        for i in range(len(num_scales)):
            seq = []
            # unsampling
            if i != 0:
                seq += [nn.Upsample(scale_factor=2, mode='nearest')]
            # if need to reduce dimension
            if num_scales[i] in reduce_dim_at:
                seq += [pad_conv_norm(cur_dim, cur_dim//2, norm_layer, activation=act_layer)]
                cur_dim = cur_dim//2
            # print ('scale {} cur_dim {}'.format(num_scales[i], cur_dim))
            # add residual blocks
            for n in range(num_resblock):
                seq += [ResnetBlock(cur_dim, norm, activation=activation)]
            # add main convolutional module
            setattr(self, 'scale_%d'%(num_scales[i]), nn.Sequential(*seq) )
            
            # add upsample module to concat with upper layers 
            # if num_scales[i] in text_upsampling_at and use_upsamle_skip:
            #     setattr(self, 'upsample_%d'%(num_scales[i]), MultiModalBlock(text_dim=cur_dim, img_dim=cur_dim//2, norm=norm, activation=activation))
            # configure side output module
            if num_scales[i] in self.side_output_at:
                setattr(self, 'tensor_to_img_%d'%(num_scales[i]), branch_out2(cur_dim))
                
        self.apply(weights_init)

        print (' downsample at {}'.format(str(reduce_dim_at)))
        
    def forward(self, sent_embeddings, z, epsilon=None):
        # sent_embeddings: [B, 1024]
        out_dict = OrderedDict()
        sent_random, kl_loss  = self.condEmbedding(sent_embeddings) # sent_random [B, 128]
        
        text = torch.cat([sent_random, z], dim=1)

        x    = self.vec_to_tensor(text)
        x_4  = self.scale_4(x)
        x_8  = self.scale_8(x_4)
        x_16 = self.scale_16(x_8)
        x_32 = self.scale_32(x_16)
        
        # skip 4x4 feature map to 32 and send to 64
        x_64 = self.scale_64(x_32)
        if 64 in self.detach_list:
            x_64 = x_64.detach()
        else:
            if 64 in self.side_output_at:
                out_dict['output_64'] = self.tensor_to_img_64(x_64)

        if self.max_output_size > 64:
            # skip 8x8 feature map to 64 and send to 128
            x_128 = self.scale_128(x_64)
            if 128 in self.detach_list:
                   x_128 = x_128.detach() 
            else:
                if 128 in self.side_output_at:
                    out_dict['output_128'] = self.tensor_to_img_128(x_128)
                    
        if self.max_output_size > 128:
            # skip 16x16 feature map to 128 and send to 256
            out_256 = self.scale_256(x_128)
            if 256 in self.side_output_at:
                out_dict['output_256'] = self.tensor_to_img_256(out_256)

        return out_dict, kl_loss
    

class Discriminator(torch.nn.Module):
    
    def __init__(self, input_size, num_chan,  hid_dim, sent_dim, emb_dim, 
                 norm='bn', disc_mode= ['global']):
        '''
        input_size can be int or list.
        num_chan: channel of generated images.
        enc_dim: Reduce images inputs to (B, enc_dim, H, W) feature
        emb_dim: The sentence embedding dimension.
        disc_mode: specify to use global, local or both. 
        '''
        super(Discriminator, self).__init__()
        self.register_buffer('device_id', torch.IntTensor(1))
        self.__dict__.update(locals())
        activ = discAct()
        norm_layer = getNormLayer(norm)

        enc_dim = hid_dim * 4 # the ImageDown output dimension

        '''user defefined'''
        if type(input_size) is int:
            if input_size==256: self.side_output_at = [64, 128, 256] 
            if input_size==128: self.side_output_at = [64, 128] 
            if input_size==64:  self.side_output_at = [64] 
        else:
            self.side_output_at = input_size
        # 64, 128, or 256 version
        self.max_output_size = max(self.side_output_at)


        _layers = []
        self.img_encoder_64   = ImageDown(64,  num_chan,  enc_dim, norm)  # 4x4
        self.pair_disc_64     = DiscClassifier(enc_dim, emb_dim, feat_size=4, norm=norm, activ=activ)
        _layers =  [nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True)]
        
        if 64 in self.side_output_at:
            self.global_img_disc_64 = nn.Sequential(*_layers)
            _layers = [nn.Linear(sent_dim, emb_dim)]
            _layers += [activ]
            self.context_emb_pipe_64 = nn.Sequential(*_layers)

        if 128 in self.side_output_at:
            self.img_encoder_128  = ImageDown(128,  num_chan, enc_dim, norm)  # 8
            self.pair_disc_128  = DiscClassifier(enc_dim, emb_dim, feat_size=4,  norm=norm, activ=activ)

            if 'local' in self.disc_mode:
                _layers = [nn.Conv2d(enc_dim, 1, kernel_size=1, padding=0, bias=True)]   # 4
                self.local_img_disc_128 = nn.Sequential(*_layers)
            if 'global' in self.disc_mode:
                _layers = [nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True)]   # 4
                self.global_img_disc_128 = nn.Sequential(*_layers)

            _layers = [nn.Linear(sent_dim, emb_dim)]
            _layers += [activ]
            self.context_emb_pipe_128 = nn.Sequential(*_layers)

        if 256 in self.side_output_at:
            self.img_encoder_256  = ImageDown(256, num_chan, enc_dim, norm)  # 8
            
            self.pair_disc_256  = DiscClassifier(enc_dim, emb_dim, feat_size=4, norm=norm, activ=activ)
            
            # shrink is used for mapping 8x8 FM to 4x4
            self.shrink = conv_norm(enc_dim, enc_dim,  norm_layer, stride=1, activation=activ, kernel_size=5, padding=0)

            if 'local' in self.disc_mode:
                _layers = [nn.Conv2d(enc_dim, 1, kernel_size=1, padding=0, bias=True)]   # 8
                self.local_img_disc_256 = nn.Sequential(*_layers)
            if 'global' in self.disc_mode:
                _layers = [nn.Conv2d(enc_dim, 1, kernel_size=4, padding=0, bias=True)]   # 1
                self.global_img_disc_256 = nn.Sequential(*_layers)

            _layers = [nn.Linear(sent_dim, emb_dim)]
            _layers += [activ]
            self.context_emb_pipe_256 = nn.Sequential(*_layers)

        self.apply(weights_init)
        print ('>> initialized a {} size discriminator'.format(self.side_output_at) )

    def forward(self, images, embdding):
        '''
        images: (B, C, H, W)
        embdding : (B, sent_dim)
        outptuts:
        -----------
        img_code B*chan*col*row
        pair_disc_out: B*1
        img_disc: B*1*col*row
        '''
        out_dict = OrderedDict()
        this_img_size = images.size()[3]
        assert this_img_size in [32, 64, 128, 256], 'wrong input size {} in image discriminator'.format(this_img_size)
        assert self.max_output_size >= this_img_size, 'image size {} exceeds expected maximum size {}'.format(this_img_size, self.max_out_size)

        img_encoder = getattr(self, 'img_encoder_{}'.format(this_img_size))
        local_img_disc    = getattr(self, 'local_img_disc_{}'.format(this_img_size), None)
        global_img_disc   = getattr(self, 'global_img_disc_{}'.format(this_img_size), None)
        pair_disc         = getattr(self, 'pair_disc_{}'.format(this_img_size))
        context_emb_pipe  = getattr(self, 'context_emb_pipe_{}'.format(this_img_size))

        sent_code = context_emb_pipe(embdding)
        img_code = img_encoder(images) 
        if this_img_size == 256:
            shrink_img_code = self.shrink(img_code)
            pair_disc_out = pair_disc(sent_code, shrink_img_code)
        else:
            pair_disc_out = pair_disc(sent_code, img_code)

        out_dict['local_img_disc']   = 1
        out_dict['global_img_disc']  = 1

        # 64 never uses local discriminator
        if 'local' in self.disc_mode and this_img_size != 64:
            local_img_disc_out          = local_img_disc(img_code) 
            out_dict['local_img_disc']  = local_img_disc_out
            
        if 'global' in self.disc_mode or this_img_size == 64:  
            global_code = shrink_img_code if this_img_size == 256 else img_code
            global_img_disc_out         = global_img_disc(global_code)
            assert global_img_disc_out.size()[3] == 1, 'global output does not equal 1x1'
            out_dict['global_img_disc'] = global_img_disc_out
            
        out_dict['pair_disc']     = pair_disc_out
        out_dict['content_code']  = None # useless

        return out_dict


class ImageDown(torch.nn.Module):
    '''
       This module encode image to 16*16 feat maps
    '''
    def __init__(self, input_size, num_chan, out_dim, norm='norm'):
        super(ImageDown, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        
        self.__dict__.update(locals())
        norm_layer = getNormLayer(norm)
        activ = discAct()
        _layers = []

        # use large kernel_size at the end to prevent zero-padding and stride
        if input_size == 64:
            cur_dim = 128 # for testing
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)] # 32
            _layers += [conv_norm(cur_dim, cur_dim*2,  norm_layer, stride=2, activation=activ)] # 16
            _layers += [conv_norm(cur_dim*2, cur_dim*4,  norm_layer, stride=2, activation=activ)] # 8
            _layers += [conv_norm(cur_dim*4, out_dim,  norm_layer, stride=1, activation=activ, kernel_size=5, padding=0)] # 4
            #_layers += [conv_norm(cur_dim*8, out_dim,  norm_layer, stride=1, activation=activ,kernel_size=3, padding=0)] # 2

        if input_size == 128:
            cur_dim = 64 # for testing
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)] # 64
            _layers += [conv_norm(cur_dim, cur_dim*2,  norm_layer, stride=2, activation=activ)] # 32
            _layers += [conv_norm(cur_dim*2, cur_dim*4,  norm_layer, stride=2, activation=activ)] # 16
            _layers += [conv_norm(cur_dim*4, cur_dim*8,  norm_layer, stride=2, activation=activ)] # 8
            _layers += [conv_norm(cur_dim*8, out_dim,  norm_layer, stride=1, activation=activ,kernel_size=5, padding=0)] # 4
        
        if input_size == 256:
            cur_dim = 32 # for testing
            _layers += [conv_norm(num_chan, cur_dim, norm_layer, stride=2, activation=activ, use_norm=False)] # 128
            _layers += [conv_norm(cur_dim, cur_dim*2,  norm_layer, stride=2, activation=activ)] # 64
            _layers += [conv_norm(cur_dim*2, cur_dim*4,  norm_layer, stride=2, activation=activ)] # 32
            _layers += [conv_norm(cur_dim*4, cur_dim*8,  norm_layer, stride=2, activation=activ)] # 16
            _layers += [conv_norm(cur_dim*8, out_dim,  norm_layer, stride=2, activation=activ)] # 8
            #_layers += [conv_norm(cur_dim*16, out_dim,  norm_layer, stride=2, activation=activ)] # 4
            
        self.node = nn.Sequential(*_layers)

    def forward(self, inputs):
        
        out = self.node(inputs)
        return out


class DiscClassifier(nn.Module):
    def __init__(self, enc_dim, emb_dim, feat_size, norm, activ):
        '''
          enc_dim: B*enc_dim*H*W
          emb_dim: the dimension of feeded embedding
          feat_size: the feature map size of the feature map. 
        '''
        super(DiscClassifier, self).__init__()
        self.__dict__.update(locals())
        norm_layer = getNormLayer(norm)
        activ = discAct()
        inp_dim = enc_dim + emb_dim
        new_feat_size = feat_size
        
        _layers =  [ conv_norm(inp_dim, enc_dim, norm_layer, kernel_size=1, stride=1, activation=activ),
                      nn.Conv2d(enc_dim, 1, kernel_size =new_feat_size, padding=0, bias=True)]
        ## _layers = [nn.Conv2d(inp_dim, 1, kernel_size=new_feat_size, padding=0, bias=True)]
        self.node = nn.Sequential(*_layers)

    def forward(self,sent_code,  img_code):
        sent_code =  sent_code.unsqueeze(-1).unsqueeze(-1)
        dst_shape = list(sent_code.size())
        #print(dst_shape, img_code.size())
        dst_shape[1] =  sent_code.size()[1]
        dst_shape[2] =  img_code.size()[2] 
        dst_shape[3] =  img_code.size()[3] 
        sent_code = sent_code.expand(dst_shape)
        #sent_code = sent_code.view(*dst_shape)
        #print(img_code.size(), sent_code.size())
        comp_inp = torch.cat([img_code, sent_code], dim=1)
        output = self.node(comp_inp)
        chn  = output.size()[1]
        output = output.view(-1, chn)

        return output



class ResnetBlock(nn.Module):
    def __init__(self, dim, norm, activation='relu', use_bias=False):
        super(ResnetBlock, self).__init__()
        norm_layer = getNormLayer(norm)
        activ = get_activation_layer(activation)
        seq = [pad_conv_norm(dim, dim, norm_layer, use_bias=use_bias, activation=activ), 
               pad_conv_norm(dim, dim, norm_layer, use_activation=False, use_bias=use_bias)]
        self.res_block = nn.Sequential(*seq)

    def forward(self, input):
        # TODO do we need to add activation? 
        # CycleGan regards this. I guess to prevent spase gradients
        
        return self.res_block(input) + input

class MultiModalBlock(nn.Module):
    def __init__(self, text_dim, img_dim, norm, activation='relu', use_bias=False, upsample_factor=3):
        super(MultiModalBlock, self).__init__()
        norm_layer = getNormLayer(norm)
        activ = get_activation_layer(activation)
        # upsampling 2^3 times
        seq = []
        cur_dim = text_dim
        for i in range(upsample_factor):
            seq += [nn.Upsample(scale_factor=2, mode='nearest')]
            seq += [pad_conv_norm(cur_dim, cur_dim//2, norm_layer, activation=activ)]
            cur_dim = cur_dim//2

        self.upsample_path = nn.Sequential(*seq)
        self.joint_path = nn.Sequential(*[
            pad_conv_norm(cur_dim+img_dim, img_dim, norm_layer, kernel_size=1, use_activation=False)
        ])
    def forward(self, text, img ):
        upsampled_text = self.upsample_path(text)
        inputs = torch.cat([img, upsampled_text],1)
        out = self.joint_path(inputs)
        return out

class Sent2FeatMap(nn.Module):
    def __init__(self, in_dim, row, col, channel, norm='bn',
                 activ = None, last_active = False):
        super(Sent2FeatMap, self).__init__()
        self.__dict__.update(locals())
        out_dim = row*col*channel
        norm_layer = getNormLayer(norm, dim=1)
        _layers = [nn.Linear(in_dim, out_dim)]
        _layers += [norm_layer(out_dim)]

        if activ is not None:
            _layers += [activ] 
        
        self.out = nn.Sequential(*_layers)    
         
    def forward(self, inputs):
        output = self.out(inputs)
        output = output.view(-1, self.channel, self.row, self.col)
        return output
