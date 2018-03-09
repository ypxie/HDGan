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

def compute_d_pair_loss(real_logit, wrong_logit, fake_logit, wgan=False):
    if wgan:
        disc = wrong_logit  + fake_logit - 2*real_logit
        return torch.mean(disc)
    else:
        real_d_loss  = torch.mean( ((real_logit) -1)**2)
        wrong_d_loss = torch.mean( ((wrong_logit))**2)
        fake_d_loss  = torch.mean( ((fake_logit))**2)

        discriminator_loss =\
            real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
        return discriminator_loss

def compute_d_img_loss(wrong_img_logit, real_img_logit, fake_logit, prob=0.5, wgan=False):
    if wgan:
        dloss = (prob*wrong_img_logit + (1-prob)*real_img_logit) - real_logit
        return torch.mean(dloss)
    else:
        wrong_d_loss = 0 if type(wrong_img_logit) in [int, float] else torch.mean( ((wrong_img_logit) -1)**2)
        real_d_loss  = 0 if type(real_img_logit) in [int, float]  else torch.mean( ((real_img_logit) -1)**2)

        real_img_d_loss = wrong_d_loss * prob + real_d_loss * (1-prob)
        fake_d_loss  = 0 if type(fake_logit) in [int, float]  else  torch.mean( ((fake_logit))**2)
        
        return fake_d_loss + real_img_d_loss

def compute_g_loss(fake_logit, wgan=False):
    if wgan:
        gloss = -fake_logit
        #gloss = -fake_img_logit
        return torch.mean(gloss)
    else:
        if type(fake_logit) in [int, float]:
            return 0
        else:
            generator_loss = torch.mean( ((fake_logit) -1)**2 )
            return generator_loss

def load_partial_state_dict(model, state_dict):
    
        own_state = model.state_dict()
        #print('own_Dict', own_state.keys(), 'state_Dict',state_dict.keys())
        for a,b in zip( own_state.keys(), state_dict.keys()):
            print(a,'_from model =====_loaded: ', b)
        for name, param in state_dict.items():
            if name is "device_id":
                pass
            else:
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

def train_gans(dataset, model_root, mode_name, netG, netD, args):
    use_img_loss   = getattr(args, 'use_img_loss', True)
    img_loss_ratio = getattr(args, 'img_loss_ratio', 1.0)
    tune_img_loss  = getattr(args, 'tune_img_loss', False)
    this_img_loss_ratio = img_loss_ratio
    print('>> using hd gan trainer')
    # helper function
    def plot_imgs(samples, epoch, typ, name, path=''):
        tmpX = save_images(samples, save=not path == '', save_path=os.path.join(path, '{}_epoch{}_{}.png'.format(name, epoch, typ)), dim_ordering='th')
        plot_img(X=tmpX, win='{}_{}.png'.format(name, typ), env=mode_name)

    def fake_sampler(bz, n ):
        x = {}
        x['output_64'] = np.random.rand(bz, 3, 64, 64)
        x['output_128'] = np.random.rand(bz, 3, 128, 128)
        x['output_256'] = np.random.rand(bz, 3, 256, 256)
        return x, x, np.random.rand(bz, 1024), None, None

    d_lr = args.d_lr
    g_lr = args.g_lr
    tot_epoch = args.maxepoch
    if not args.debug_mode:
        train_sampler = dataset.train.next_batch
        test_sampler  = dataset.test.next_batch
        number_example = dataset.train._num_examples
        updates_per_epoch = int(number_example / args.batch_size)
    else:
        train_sampler = fake_sampler
        test_sampler = fake_sampler
        number_example = 16
        updates_per_epoch = 10

    ''' configure optimizer '''
    num_test_forward = 1 # 64 // args.batch_size // args.test_sample_num # number of testing samples to show
    if args.wgan:
        optimizerD = optim.RMSprop(netD.parameters(), lr= d_lr,  weight_decay=args.weight_decay)
        optimizerG = optim.RMSprop(netG.parameters(), lr= g_lr,  weight_decay=args.weight_decay)
    else:
        optimizerD = optim.Adam(netD.parameters(), lr= d_lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
        optimizerG = optim.Adam(netG.parameters(), lr= g_lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)

    model_folder = os.path.join(model_root, mode_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    plot_save_path = os.path.join(model_folder, 'plot_save.pth')
    plot_dict = {'disc':[], 'gen':[]}

    ''' load model '''
    if args.reuse_weights :
        D_weightspath = os.path.join(model_folder, 'D_epoch{}.pth'.format(args.load_from_epoch))
        G_weightspath = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))

        if os.path.exists(D_weightspath) and os.path.exists(G_weightspath):

            #assert os.path.exists(D_weightspath) and os.path.exists(G_weightspath)
            weights_dict = torch.load(D_weightspath, map_location=lambda storage, loc: storage)
            print('reload weights from {}'.format(D_weightspath))
            load_partial_state_dict(netD, weights_dict)
            # netD.load_state_dict(weights_dict)# 12)
            print('reload weights from {}'.format(G_weightspath))
            weights_dict = torch.load(G_weightspath, map_location=lambda storage, loc: storage)
            load_partial_state_dict(netG, weights_dict)
            # netG.load_state_dict(weights_dict)# 12)

            start_epoch = args.load_from_epoch + 1
            if os.path.exists(plot_save_path):
                plot_dict = torch.load(plot_save_path)
        else:
            print ('{} or {} do not exist!!'.format(D_weightspath, G_weightspath))
            raise NotImplementedError
    else:
        start_epoch = 1

    d_loss_plot = plot_scalar(name = "d_loss", env= mode_name, rate = args.display_freq)
    g_loss_plot = plot_scalar(name = "g_loss", env= mode_name, rate = args.display_freq)
    content_loss_plot = plot_scalar(name = "content_loss", env= mode_name, rate = args.display_freq)
    lr_plot = plot_scalar(name = "lr", env= mode_name, rate = args.display_freq)

    all_keys = ["output_64", "output_128", "output_256"]
    g_plot_dict, d_plot_dict = {}, {}
    for this_key in all_keys:
        g_plot_dict[this_key] = plot_scalar(name = "g_img_loss_" + this_key, env= mode_name, rate = args.display_freq)
        d_plot_dict[this_key] = plot_scalar(name = "d_img_loss_" + this_key, env= mode_name, rate = args.display_freq)

    z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    z = to_device(z, netG.device_id, requires_grad=False)
    # test the fixed image for every epoch
    fixed_images, _, fixed_embeddings, _, _ = test_sampler(args.batch_size, 1)
    fixed_embeddings = to_device(fixed_embeddings, netG.device_id, volatile=True)
    fixed_z_data = [torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1) for _ in range(args.test_sample_num)]
    fixed_z_list = [to_device(a, netG.device_id, volatile=True) for a in fixed_z_data] # what?


    global_iter = 0
    gen_iterations = 0
    last_ncritic = 0
    for epoch in range(start_epoch, tot_epoch):
        start_timer = time.time()
        # learning rate
        if epoch % args.epoch_decay == 0:
            d_lr = d_lr/2
            g_lr = g_lr/2
            if tune_img_loss:
                this_img_loss_ratio = this_img_loss_ratio/2
                print('this_img_loss_ratio: ', this_img_loss_ratio)

            set_lr(optimizerD, d_lr)
            set_lr(optimizerG, g_lr)

        for it in range(updates_per_epoch):
            netG.train()
            if epoch <= args.ncritic_epoch_range:
                if (epoch < 2) and (gen_iterations < 100 or (gen_iterations < 1000 and gen_iterations % 20 == 0))  :
                    ncritic = 5
                    #print ('>> set ncritic to {}'.format(ncritic))
                elif gen_iterations % 50 == 0:
                    ncritic = 10    
                    #print ('>> set ncritic to {}'.format(ncritic))
                else:
                    ncritic = args.ncritic
                #print ('>> set ncritic to {}'.format(ncritic))
            else:
                ncritic = args.ncritic

            if last_ncritic != ncritic:
                print ('change ncritic {} -> {}'.format(last_ncritic, ncritic))
                last_ncritic = ncritic

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
                d_loss_val_dict = {}
                for key, _ in fake_images.items():
                    # iterate over image of different sizes.
                    this_img   = to_device(images[key], netD.device_id)
                    this_wrong = to_device(wrong_images[key], netD.device_id)
                    this_fake  = Variable(fake_images[key].data) # to cut connection to netG

                    real_dict   = netD(this_img,   embeddings)
                    wrong_dict  = netD(this_wrong, embeddings)
                    fake_dict   = netD(this_fake,  embeddings)
                    real_logit,  real_img_logit_local,  real_img_logit_global  =  real_dict['pair_disc'], real_dict['local_img_disc'], real_dict['global_img_disc']
                    wrong_logit, wrong_img_logit_local, wrong_img_logit_global =  wrong_dict['pair_disc'], wrong_dict['local_img_disc'], wrong_dict['global_img_disc']
                    fake_logit,  fake_img_logit_local,  fake_img_logit_global  =  fake_dict['pair_disc'], fake_dict['local_img_disc'], fake_dict['global_img_disc']

                    # compute loss
                    #chose_img_real = wrong_img_logit if random.random() > 0.1 else real_img_logit
                    discriminator_loss += compute_d_pair_loss(real_logit, wrong_logit, fake_logit, args.wgan)
                    if use_img_loss:
                        local_loss  = compute_d_img_loss(wrong_img_logit_local,  real_img_logit_local,   fake_img_logit_local,  prob=0.5, wgan=args.wgan )
                        global_loss = compute_d_img_loss(wrong_img_logit_global, real_img_logit_global,  fake_img_logit_global, prob=0.5, wgan=args.wgan )
                        if type(local_loss) in [int, float] or type(global_loss) in [int, float]: # one of them is int
                            img_loss = local_loss + global_loss
                        else:
                            img_loss = (local_loss + global_loss)*0.5
                        
                        discriminator_loss +=  this_img_loss_ratio * img_loss 
                        d_plot_dict[key].plot(img_loss.cpu().data.numpy().mean())
                    #else:
                    #    print('Hey, ya are not using img loss in disc')      
                d_loss_val  = discriminator_loss.cpu().data.numpy().mean()
                d_loss_val = -d_loss_val if args.wgan else d_loss_val
                discriminator_loss.backward()
                optimizerD.step()
                netD.zero_grad()
                d_loss_plot.plot(d_loss_val)
                plot_dict['disc'].append(d_loss_val)

            ''' update G '''
            for p in netD.parameters(): p.requires_grad = False  # to avoid computation
            netG.zero_grad()
           
            z.data.normal_(0, 1) # resample random noises
            fake_images, kl_loss = netG(embeddings, z)

            loss_val  = 0
            generator_loss = args.KL_COE*kl_loss
            for key, _ in fake_images.items():
                # iterate over image of different sizes.
                this_fake  = fake_images[key]
                fake_dict  = netD(this_fake,  embeddings)
                fake_pair_logit, fake_img_logit_local, fake_img_logit_global, fake_img_code  = \
                fake_dict['pair_disc'], fake_dict['local_img_disc'], fake_dict['global_img_disc'], fake_dict['content_code']
                generator_loss += compute_g_loss(fake_pair_logit, args.wgan)
                
                if use_img_loss:
                    
                    local_loss  = compute_g_loss(fake_img_logit_local, args.wgan)
                    global_loss = compute_g_loss(fake_img_logit_global, args.wgan)
                    if type(local_loss) in [int, float] or type(global_loss) in [int, float]: # one of them is int
                        img_loss_ = local_loss + global_loss
                    else:
                        img_loss_ = (local_loss + global_loss)*0.5
                    
                    generator_loss += img_loss_ * this_img_loss_ratio
                    g_plot_dict[key].plot(img_loss_.cpu().data.numpy().mean())

                #else:
                #    print('Hey, ya are not using img loss in generator')        
            generator_loss.backward()
            g_loss_val = generator_loss.cpu().data.numpy().mean()

            optimizerG.step()
            netG.zero_grad()
            g_loss_plot.plot(g_loss_val)
            lr_plot.plot(g_lr)
            plot_dict['gen'].append(g_loss_val)
            gen_iterations += 1
            global_iter += 1
            # visualize train samples
            if it % args.verbose_per_iter == 0:
                for k, sample in fake_images.items():
                    # plot_imgs(sample.cpu().data.numpy(), epoch, k, 'train_samples')
                    plot_imgs([images[k], sample.cpu().data.numpy()], epoch, k, 'train_images')
                print ('[epoch %d/%d iter %d]: lr = %.6f g_loss = %.5f d_loss= %.5f' % (epoch, tot_epoch, it, g_lr, g_loss_val, d_loss_val))
                sys.stdout.flush()

        ''' visualize test per epoch '''
        # generate samples
        gen_samples = []
        img_samples = []
        vis_samples = {}
        for idx_test in range(num_test_forward + 1):
            #sent_emb_test, _ =  netG.condEmbedding(test_embeddings)
            if idx_test == 0:
                test_images, test_embeddings = fixed_images, fixed_embeddings

            else:
                test_images, _, test_embeddings, _, _ = test_sampler(args.batch_size, 1)
                test_embeddings = to_device(test_embeddings, netG.device_id, volatile=True)
                testing_z = Variable(z.data, volatile=True)

            tmp_samples = {}

            for t in range(args.test_sample_num):

                if idx_test == 0: # plot fixed
                    testing_z = fixed_z_list[t]
                else:
                    testing_z.data.normal_(0, 1)

                samples, _ = netG(test_embeddings, testing_z)

                if idx_test == 0 and t == 0:
                    for k in samples.keys():
                        vis_samples[k] = [None for i in range(args.test_sample_num + 1)] # +1 to fill real image

                # Oops! very tricky to organize data for plot inputs!!!
                # vis_samples[k] = [real data, sample1, sample2, sample3, ... sample_args.test_sample_num]
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
        # visualize samples
        for typ, v in vis_samples.items():
            plot_imgs(v, epoch, typ, 'test_samples', path=model_folder)

        # save weights
        try:
            if epoch % args.save_freq == 0:
                netD = netD.cpu()
                netG = netG.cpu()
                torch.save(netD.state_dict(), os.path.join(model_folder, 'D_epoch{}.pth'.format(epoch)))
                torch.save(netG.state_dict(), os.path.join(model_folder, 'G_epoch{}.pth'.format(epoch)))
                print('save weights at {}'.format(model_folder))
                torch.save(plot_dict, plot_save_path)
                netD = netD.cuda(args.device_id)
                netG = netG.cuda(args.device_id)
            print ('epoch {}/{} finished [time = {}s] ...'.format(epoch, tot_epoch, end_timer))
        except:
            print('Failed to save model, will try next time')
            