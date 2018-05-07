import numpy as np
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter

from torch.nn.utils import clip_grad_norm
from .proj_utils.plot_utils import *
from .proj_utils.torch_utils import *
import time
import json
import functools 


def to_img_dict_(*inputs, super512=False):
    if type(inputs[0]) == tuple:
        inputs = inputs[0]
    res = {}
    res['output_64'] = inputs[0]
    res['output_128'] = inputs[1]
    res['output_256'] = inputs[2]
    # generator returns different things for 512HDGAN
    if not super512:
        # from Generator
        mean_var = (inputs[3], inputs[4])
        loss = mean_var
    else:
        # from GeneratorL1Loss of 512HDGAN
        res['output_512'] = inputs[3]
        l1loss = inputs[4] # l1 loss
        loss = l1loss

    return res, loss

def get_KL_Loss(mu, logvar):
    kld = mu.pow(2).add(logvar.mul(2).exp()).add(-1).mul(0.5).add(logvar.mul(-1))
    kl_loss = torch.mean(kld)
    return kl_loss

def compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels):

    criterion = nn.MSELoss()
    real_d_loss = criterion(real_logit, real_labels)
    wrong_d_loss = criterion(wrong_logit, fake_labels)
    fake_d_loss = criterion(fake_logit, fake_labels)

    discriminator_loss =\
        real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
    return discriminator_loss

def compute_d_img_loss(wrong_img_logit, real_img_logit, fake_img_logit, real_labels, fake_labels):

    criterion = nn.MSELoss()
    wrong_d_loss = criterion(wrong_img_logit, real_labels)
    real_d_loss  = criterion(real_img_logit, real_labels)
    fake_d_loss  = criterion(fake_img_logit, fake_labels)

    return fake_d_loss + (wrong_d_loss+real_d_loss) / 2

def compute_g_loss(fake_logit, real_labels):
    criterion = nn.MSELoss()
    generator_loss = criterion(fake_logit, real_labels)
    return generator_loss

def plot_imgs(samples, epoch, typ, name, path='', model_name=None):
    tmpX = save_images(samples, save=not path == '', save_path=os.path.join(
        path, '{}_epoch{}_{}.png'.format(name, epoch, typ)), dim_ordering='th')
    plot_img(X=tmpX, win='{}_{}.png'.format(name, typ), env=model_name)


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

    d_lr = args.d_lr
    g_lr = args.g_lr
    tot_epoch = args.maxepoch

    ''' get train and test data sampler '''
    train_sampler = iter(dataset[0])
    test_sampler = iter(dataset[1])

    updates_per_epoch = int(dataset[0]._num_examples / args.batch_size)

    ''' configure optimizer '''
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(0.5, 0.999))

    model_folder = os.path.join(model_root, model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    #-------------load model from  checkpoint---------------------------#
    if args.reuse_weights:
        D_weightspath = os.path.join(
            model_folder, 'D_epoch{}.pth'.format(args.load_from_epoch))
        G_weightspath = os.path.join(
            model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))

        if os.path.exists(D_weightspath) and os.path.exists(G_weightspath):
            weights_dict = torch.load(
                D_weightspath, map_location=lambda storage, loc: storage)
            print('reload weights from {}'.format(D_weightspath))
            netD_ = netD.module if 'DataParallel' in str(type(netD)) else netD
            netD_.load_state_dict(weights_dict, strict=False)

            print('reload weights from {}'.format(G_weightspath))
            weights_dict = torch.load(
                G_weightspath, map_location=lambda storage, loc: storage)
            netG_ = netG.module if 'DataParallel' in str(type(netG)) else netG
            netG_.load_state_dict(weights_dict, strict=False)

            start_epoch = args.load_from_epoch + 1
            d_lr /= 2 ** (start_epoch // args.epoch_decay)
            g_lr /= 2 ** (start_epoch // args.epoch_decay) 
        else:
            raise ValueError('{} or {} do not exist'.format(D_weightspath, G_weightspath))
    else:
        start_epoch = 1

    #-------------init ploters for losses----------------------------#
    d_loss_plot = plot_scalar(
        name="d_loss", env=model_name, rate=args.display_freq)
    g_loss_plot = plot_scalar(
        name="g_loss", env=model_name, rate=args.display_freq)
    lr_plot = plot_scalar(name="lr", env=model_name, rate=args.display_freq)
    kl_loss_plot = plot_scalar(name="kl_loss", env=model_name, rate=args.display_freq)
    
    all_keys = ["output_64", "output_128", "output_256"]
    g_plot_dict, d_plot_dict = {}, {}
    for this_key in all_keys:
        g_plot_dict[this_key] = plot_scalar(
            name="g_img_loss_" + this_key, env=model_name, rate=args.display_freq)
        d_plot_dict[this_key] = plot_scalar(
            name="d_img_loss_" + this_key, env=model_name, rate=args.display_freq)

    #--------Generator niose placeholder used for testing------------#
    z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    z = to_device(z)
    fixed_images, _, fixed_embeddings, _, _ = next(test_sampler)
    fixed_embeddings = to_device(fixed_embeddings)
    fixed_z_data = [torch.FloatTensor(args.batch_size, args.noise_dim).normal_(
        0, 1) for _ in range(args.test_sample_num)]
    fixed_z_list = [to_device(a) for a in fixed_z_data]

    # create discrimnator label placeholder (not a good way)
    REAL_global_LABELS = Variable(torch.FloatTensor(args.batch_size, 1).fill_(1)).cuda()
    FAKE_global_LABELS = Variable(torch.FloatTensor(args.batch_size, 1).fill_(0)).cuda()
    REAL_local_LABELS = Variable(torch.FloatTensor(args.batch_size, 1, 4, 4).fill_(1)).cuda()
    FAKE_local_LABELS = Variable(torch.FloatTensor(args.batch_size, 1, 4, 4).fill_(0)).cuda()

    def get_labels(logit):
        # get discriminator labels for real and fake
        if logit.size(-1) == 1: 
            return REAL_global_LABELS.view_as(logit), FAKE_global_LABELS.view_as(logit)
        else:
            return REAL_local_LABELS.view_as(logit), FAKE_local_LABELS.view_as(logit)

    to_img_dict = functools.partial(to_img_dict_, super512=args.finest_size == 512)
    #--------Start training------------#
    for epoch in range(start_epoch, tot_epoch):
        start_timer = time.time()
        '''decay learning rate every epoch_decay epoches'''
        if epoch % args.epoch_decay == 0:
            d_lr = d_lr/2
            g_lr = g_lr/2

            set_lr(optimizerD, d_lr)
            set_lr(optimizerG, g_lr)

        # reset to prevent StopIteration
        train_sampler = iter(dataset[0]) 
        test_sampler = iter(dataset[1])

        netG.train()
        netD.train()
        for it in range(updates_per_epoch):
            ncritic = args.ncritic

            for _ in range(ncritic):
                ''' Sample data '''
                try:
                    images, wrong_images, np_embeddings, _, _ = next(train_sampler)
                except:
                    train_sampler = iter(dataset[0]) # reset
                    images, wrong_images, np_embeddings, _, _ = next(train_sampler)
                    
                embeddings = to_device(np_embeddings)
                z.data.normal_(0, 1)

                ''' update D '''
                for p in netD.parameters(): p.requires_grad = True
                netD.zero_grad()

                fake_images, mean_var = to_img_dict(netG(embeddings, z))

                discriminator_loss = 0
                ''' iterate over image of different sizes.'''
                for key, _ in fake_images.items():
                    this_img = to_device(images[key])
                    this_wrong = to_device(wrong_images[key])
                    this_fake = Variable(fake_images[key].data)

                    real_logit,  real_img_logit_local = netD(this_img, embeddings)
                    wrong_logit, wrong_img_logit_local = netD(this_wrong, embeddings)
                    fake_logit,  fake_img_logit_local = netD(this_fake, embeddings)

                    ''' compute disc pair loss '''
                    real_labels, fake_labels = get_labels(real_logit)
                    pair_loss =  compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels)

                    ''' compute disc image loss '''
                    real_labels, fake_labels = get_labels(real_img_logit_local)
                    img_loss = compute_d_img_loss(wrong_img_logit_local, real_img_logit_local, fake_img_logit_local, real_labels, fake_labels)

                    discriminator_loss += (pair_loss + img_loss)

                    d_plot_dict[key].plot(to_numpy(img_loss).mean())

                d_loss_val = to_numpy(discriminator_loss).mean()
                discriminator_loss.backward()

                optimizerD.step()
                netD.zero_grad()

                d_loss_plot.plot(d_loss_val)

            ''' update G '''
            for p in netD.parameters(): p.requires_grad = False  # to avoid computation
            netG.zero_grad()

            # TODO Test if we do need to sample again in Birds and Flowers
            # z.data.normal_(0, 1)  # resample random noises
            # fake_images, kl_loss = netG(embeddings, z)

            loss_val = 0
            if type(mean_var) == tuple:
                kl_loss = get_KL_Loss(mean_var[0], mean_var[1])
                kl_loss_val = to_numpy(kl_loss).mean()
                generator_loss = args.KL_COE * kl_loss
            else:
                # when trian 512HDGAN. KL loss is fixed.
                # Here we optimize pixel-wise l1 loss
                generator_loss = mean_var

            kl_loss_plot.plot(kl_loss_val)
            #---- iterate over image of different sizes ----#
            '''Compute gen loss'''
            for key, _ in fake_images.items():
                this_fake = fake_images[key]
                fake_pair_logit, fake_img_logit_local = netD(this_fake, embeddings)

                # -- compute pair loss ---
                real_labels, _ = get_labels(fake_pair_logit)
                generator_loss += compute_g_loss(fake_pair_logit, real_labels)

                # -- compute image loss ---
                real_labels, _ = get_labels(fake_img_logit_local)
                img_loss = compute_g_loss(fake_img_logit_local, real_labels)
                generator_loss += img_loss
                g_plot_dict[key].plot(to_numpy(img_loss).mean())

            generator_loss.backward()
            g_loss_val = to_numpy(generator_loss).mean()

            optimizerG.step()
            netG.zero_grad()
            g_loss_plot.plot(g_loss_val)
            lr_plot.plot(g_lr)

            # --- visualize train samples----
            if it % args.verbose_per_iter == 0:
                for k, sample in fake_images.items():
                    plot_imgs([to_numpy(images[k]), to_numpy(sample)],
                              epoch, k, 'train_images', model_name=model_name)
                print('[epoch %d/%d iter %d/%d]: lr = %.6f g_loss = %.5f d_loss= %.5f' %
                      (epoch, tot_epoch, it, updates_per_epoch, g_lr, g_loss_val, d_loss_val))
                sys.stdout.flush()

        # generate and visualize testing results per epoch
        vis_samples = {}
        # display original image and the sampled images
        for idx_test in range(2):
            if idx_test == 0:
                test_images, test_embeddings = fixed_images, fixed_embeddings
            else:
                test_images, _, test_embeddings, _, _ = next(test_sampler)
                test_embeddings = to_device(test_embeddings)
                testing_z = Variable(z.data, volatile=True)
            tmp_samples = {}
            for t in range(args.test_sample_num):
                if idx_test == 0:
                    testing_z = fixed_z_list[t]
                else:
                    testing_z.data.normal_(0, 1)
                fake_images, _ = to_img_dict(netG(test_embeddings, testing_z))
                samples = fake_images
                if idx_test == 0 and t == 0:
                    for k in samples.keys():
                        #  +1 to make space for real image
                        vis_samples[k] = [None for i in range(
                            args.test_sample_num + 1)]

                for k, v in samples.items():
                    cpu_data = to_numpy(v) 
                    if t == 0:
                        if vis_samples[k][0] is None:
                            vis_samples[k][0] = test_images[k]
                        else:
                            vis_samples[k][0] = np.concatenate(
                                [vis_samples[k][0], test_images[k]], 0)

                    if vis_samples[k][t+1] is None:
                        vis_samples[k][t+1] = cpu_data
                    else:
                        vis_samples[k][t+1] = np.concatenate(
                            [vis_samples[k][t+1], cpu_data], 0)

        end_timer = time.time() - start_timer
        # visualize testing samples
        for typ, v in vis_samples.items():
            plot_imgs(v, epoch, typ, 'test_samples', path=model_folder, model_name=model_name)

        ''' save weights '''
        if epoch % args.save_freq == 0:
            netD = netD.cpu()
            netG = netG.cpu()
            netD_ = netD.module if 'DataParallel' in str(type(netD)) else netD
            netG_ = netG.module if 'DataParallel' in str(type(netD)) else netG
            torch.save(netD_.state_dict(), os.path.join(
                model_folder, 'D_epoch{}.pth'.format(epoch)))
            torch.save(netG_.state_dict(), os.path.join(
                model_folder, 'G_epoch{}.pth'.format(epoch)))
            print('save weights at {}'.format(model_folder))
            netD = netD.cuda()
            netG = netG.cuda()
        print(
            'epoch {}/{} finished [time = {}s] ...'.format(epoch, tot_epoch, end_timer))


