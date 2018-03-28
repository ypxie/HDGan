import numpy as np
import os, sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from   torch import autograd
from   torch.autograd import Variable
from   torch.nn import Parameter

from  torch.nn.utils import clip_grad_norm
from .proj_utils.plot_utils import *
from .proj_utils.torch_utils import *
import time, json
TINY = 1e-8


def train_gans(dataset, model_root, model_name, netG, netD, args):
    """
    Parameters:
    ----------
    dataset: 
        data loader. refers to fuel.dataset
    model_root: 
        the folder to save the model weights
    model_name : 
        the model_name 
    netG:
        Generator
    netD:
        Descriminator
    """

    img_loss_ratio = 1.0   # the weight of img_loss. 
    d_lr = args.d_lr
    g_lr = args.g_lr
    tot_epoch = args.maxepoch

    ''' get train and test data sampler '''
    train_sampler = dataset.train.next_batch
    test_sampler  = dataset.test.next_batch
    number_example = dataset.train._num_examples
    updates_per_epoch = int(number_example / args.batch_size)
    
    
    ''' configure optimizer '''
    optimizerD = optim.Adam(netD.parameters(), lr= d_lr, betas=(0.5, 0.999) )
    optimizerG = optim.Adam(netG.parameters(), lr= g_lr, betas=(0.5, 0.999) )

    model_folder = os.path.join(model_root, model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    #-------------load model from  checkpoint---------------------------#
    if args.reuse_weights :
        D_weightspath = os.path.join(model_folder, 'D_epoch{}.pth'.format(args.load_from_epoch))
        G_weightspath = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))

        if os.path.exists(D_weightspath) and os.path.exists(G_weightspath):
            weights_dict = torch.load(D_weightspath, map_location=lambda storage, loc: storage)
            print('reload weights from {}'.format(D_weightspath))
            load_partial_state_dict(netD, weights_dict)

            print('reload weights from {}'.format(G_weightspath))
            weights_dict = torch.load(G_weightspath, map_location=lambda storage, loc: storage)
            load_partial_state_dict(netG, weights_dict)

            start_epoch = args.load_from_epoch + 1
            
        else:
            print ('{} or {} do not exist!!'.format(D_weightspath, G_weightspath))
            raise NotImplementedError
    else:
        start_epoch = 1
    #----------------------------------------------------------------#

    #-------------init ploters for losses----------------------------#

    d_loss_plot = plot_scalar(name = "d_loss", env= model_name, rate = args.display_freq)
    g_loss_plot = plot_scalar(name = "g_loss", env= model_name, rate = args.display_freq)
    lr_plot = plot_scalar(name = "lr", env= model_name, rate = args.display_freq)

    all_keys = ["output_64", "output_128", "output_256"]
    g_plot_dict, d_plot_dict = {}, {}
    for this_key in all_keys:
        g_plot_dict[this_key] = plot_scalar(name = "g_img_loss_" + this_key, env= model_name, rate = args.display_freq)
        d_plot_dict[this_key] = plot_scalar(name = "d_img_loss_" + this_key, env= model_name, rate = args.display_freq)
    #---------------------------------------------------------------#
    
    #--------Generator niose placeholder used for testing------------# 
    z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    z = to_device(z, netG.device_id, requires_grad=False)
    fixed_images, _, fixed_embeddings, _, _ = test_sampler(args.batch_size, 1)
    fixed_embeddings = to_device(fixed_embeddings, netG.device_id, volatile=True)
    fixed_z_data = [torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1) for _ in range(args.test_sample_num)]
    fixed_z_list = [to_device(a, netG.device_id, volatile=True) for a in fixed_z_data] 
    #---------------------------------------------------------------#


    for epoch in range(start_epoch, tot_epoch):
        start_timer = time.time()
        '''decay learning rate every epoch_decay epoches'''
        if epoch % args.epoch_decay == 0:
            d_lr = d_lr/2
            g_lr = g_lr/2
            
            set_lr(optimizerD, d_lr)
            set_lr(optimizerG, g_lr)

        for it in range(updates_per_epoch):
            netG.train()
            ncritic = args.ncritic

            for _ in range(ncritic):
                ''' Sample data '''
                images, wrong_images, np_embeddings, _, _ = train_sampler(args.batch_size, args.num_emb)
                embeddings = to_device(np_embeddings, netD.device_id, requires_grad=False)
                z.data.normal_(0, 1)

                ''' update D '''
                for p in netD.parameters(): p.requires_grad = True
                netD.zero_grad()

                g_emb = Variable(embeddings.data, volatile=True)
                g_z = Variable(z.data , volatile=True)
                fake_images, _ = netG(g_emb, g_z)

                discriminator_loss = 0
                ''' iterate over image of different sizes.'''
                for key, _ in fake_images.items():
                    this_img   = to_device(images[key], netD.device_id)
                    this_wrong = to_device(wrong_images[key], netD.device_id)
                    this_fake  = Variable(fake_images[key].data) 

                    real_dict   = netD(this_img,   embeddings)
                    wrong_dict  = netD(this_wrong, embeddings)
                    fake_dict   = netD(this_fake,  embeddings)
                    real_logit,  real_img_logit_local  =  real_dict['pair_disc'],  real_dict['local_img_disc']
                    wrong_logit, wrong_img_logit_local =  wrong_dict['pair_disc'], wrong_dict['local_img_disc']
                    fake_logit,  fake_img_logit_local  =  fake_dict['pair_disc'],  fake_dict['local_img_disc']
                    
                    ''' compute pair loss '''
                    discriminator_loss += compute_d_pair_loss(real_logit, wrong_logit, fake_logit)
                    ''' compute image loss '''
                    img_loss  = compute_d_img_loss(wrong_img_logit_local,  real_img_logit_local,   fake_img_logit_local,  prob=0.5)

                    discriminator_loss +=  img_loss_ratio * img_loss 
                    d_plot_dict[key].plot(img_loss.cpu().data.numpy().mean()) 
                
                d_loss_val  = discriminator_loss.cpu().data.numpy().mean()
                discriminator_loss.backward()
                optimizerD.step()
                netD.zero_grad()
                d_loss_plot.plot(d_loss_val)


            ''' update G '''
            for p in netD.parameters(): p.requires_grad = False  # to avoid computation
            netG.zero_grad()
           
            z.data.normal_(0, 1) # resample random noises
            fake_images, kl_loss = netG(embeddings, z)

            loss_val  = 0
            generator_loss = args.KL_COE*kl_loss

            #---- iterate over image of different sizes ----#
            for key, _ in fake_images.items():
                this_fake  = fake_images[key]
                fake_dict  = netD(this_fake,  embeddings)
                fake_pair_logit, fake_img_logit_local = fake_dict['pair_disc'], fake_dict['local_img_disc']
                
                #-- compute pair loss ---
                generator_loss += compute_g_loss(fake_pair_logit)
                #-- compute image loss ---
                img_loss_  = compute_g_loss(fake_img_logit_local)
                
                generator_loss += img_loss_ * img_loss_ratio
                g_plot_dict[key].plot(img_loss_.cpu().data.numpy().mean())

            generator_loss.backward()
            g_loss_val = generator_loss.cpu().data.numpy().mean()

            optimizerG.step()
            netG.zero_grad()
            g_loss_plot.plot(g_loss_val)
            lr_plot.plot(g_lr)

            # --- visualize train samples----
            if it % args.verbose_per_iter == 0:
                for k, sample in fake_images.items():
                    plot_imgs([images[k], sample.cpu().data.numpy()], epoch, k, 'train_images', model_name=model_name)
                print ('[epoch %d/%d iter %d]: lr = %.6f g_loss = %.5f d_loss= %.5f' % (epoch, tot_epoch, it, g_lr, g_loss_val, d_loss_val))
                sys.stdout.flush()
                
        # generate and visualize testing results per epoch
        vis_samples = {}
        # display original image and the sampled images 
        for idx_test in range(2): 
            if idx_test == 0:
                test_images, test_embeddings = fixed_images, fixed_embeddings
            else:
                test_images, _, test_embeddings, _, _ = test_sampler(args.batch_size, 1)
                test_embeddings = to_device(test_embeddings, netG.device_id, volatile=True)
                testing_z = Variable(z.data, volatile=True)
            tmp_samples = {}
            for t in range(args.test_sample_num):
                if idx_test == 0: 
                    testing_z = fixed_z_list[t]
                else:
                    testing_z.data.normal_(0, 1)
                samples, _ = netG(test_embeddings, testing_z)
                if idx_test == 0 and t == 0:
                    for k in samples.keys():
                        #  +1 to make space for real image 
                        vis_samples[k] = [None for i in range(args.test_sample_num + 1)] 

                for k, v in samples.items():
                    cpu_data = v.cpu().data.numpy()
                    if t == 0:
                        if vis_samples[k][0] is None:
                            vis_samples[k][0] = test_images[k]
                        else:
                            vis_samples[k][0] =  np.concatenate([ vis_samples[k][0], test_images[k]], 0)

                    if vis_samples[k][t+1] is None:
                        vis_samples[k][t+1] = cpu_data
                    else:
                        vis_samples[k][t+1] = np.concatenate([vis_samples[k][t+1], cpu_data], 0)

        end_timer = time.time() - start_timer
        # visualize testing samples
        for typ, v in vis_samples.items():
            plot_imgs(v, epoch, typ, 'test_samples', path=model_folder, model_name=model_name)

        ''' save weights '''
        if epoch % args.save_freq == 0:
            netD = netD.cpu()
            netG = netG.cpu()
            torch.save(netD.state_dict(), os.path.join(model_folder, 'D_epoch{}.pth'.format(epoch)))
            torch.save(netG.state_dict(), os.path.join(model_folder, 'G_epoch{}.pth'.format(epoch)))
            print('save weights at {}'.format(model_folder))
            
            netD = netD.cuda(args.device_id)
            netG = netG.cuda(args.device_id)
        print ('epoch {}/{} finished [time = {}s] ...'.format(epoch, tot_epoch, end_timer))

def compute_d_pair_loss(real_logit, wrong_logit, fake_logit):
    
    real_d_loss  = torch.mean( ((real_logit) -1)**2)
    wrong_d_loss = torch.mean( ((wrong_logit))**2)
    fake_d_loss  = torch.mean( ((fake_logit))**2)

    discriminator_loss =\
        real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
    return discriminator_loss

def compute_d_img_loss(wrong_img_logit, real_img_logit, fake_logit, prob=0.5):
    
    wrong_d_loss = torch.mean( ((wrong_img_logit) -1)**2)
    real_d_loss  = torch.mean( ((real_img_logit) -1)**2)

    real_img_d_loss = wrong_d_loss * prob + real_d_loss * (1-prob)
    fake_d_loss  =  torch.mean( ((fake_logit))**2)
    
    return fake_d_loss + real_img_d_loss

def compute_g_loss(fake_logit):
    generator_loss = torch.mean( ((fake_logit) -1)**2 )
    return generator_loss

def plot_imgs(samples, epoch, typ, name, path='', model_name=None):
    tmpX = save_images(samples, save=not path == '', save_path=os.path.join(path, '{}_epoch{}_{}.png'.format(name, epoch, typ)), dim_ordering='th')
    plot_img(X=tmpX, win='{}_{}.png'.format(name, typ), env=model_name)



def load_partial_state_dict(model, state_dict):
    
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            raise KeyError('unexpected key "{}" in state_dict'
                            .format(name))
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except:
            print('While copying the parameter named {}, whose dimensions in the model are'
                    ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                        name, own_state[name].size(), param.size()))
            raise
    print ('>> load partial state dict: {} initialized'.format(len(state_dict)))

