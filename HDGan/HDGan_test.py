import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter

from torch.nn.utils import clip_grad_norm
from .proj_utils.plot_utils import *
from .proj_utils.local_utils import *
from .proj_utils.torch_utils import *
from .HDGan import to_img_dict_
from PIL import Image, ImageDraw, ImageFont
import functools
import time, json, h5py

def test_gans(dataset, model_root, mode_name, save_root , netG,  args):
    # helper function
    netG.eval()

    test_sampler = dataset.next_batch_test
    highest_res = dataset.imsize
    
    model_folder = os.path.join(model_root, mode_name)
    model_marker = mode_name + '_G_epoch_{}'.format(args.load_from_epoch)

    save_folder  = os.path.join(save_root, model_marker )   # to be defined in the later part
    save_h5    = os.path.join(save_root, model_marker+'.h5')
    org_h5path = os.path.join(save_root, 'original.h5')
    mkdirs(save_folder)
    
    ''' load model '''
    assert args.load_from_epoch != '', 'args.load_from_epoch is empty'
    G_weightspath = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))
    print('reload weights from {}'.format(G_weightspath))
    weights_dict = torch.load(G_weightspath, map_location=lambda storage, loc: storage)
    #load_partial_state_dict(netG, weights_dict)
    sample_weight_name = [a for a in weights_dict.keys()][0]
    if 'module' in sample_weight_name: # if the saved model is wrapped by DataParallel. 
        netG = nn.parallel.DataParallel(netG, device_ids=[0])
    ## TODO note that strict is set to false for now. It is a bit risky
    netG.load_state_dict(weights_dict, strict=False)

    testing_z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    testing_z = to_device(testing_z)

    num_examples = dataset._num_examples
    total_number = num_examples * args.test_sample_num

    all_choosen_caption = []
    org_file_not_exists = not os.path.exists(org_h5path)

    if org_file_not_exists:
        org_h5 = h5py.File(org_h5path,'w')
        org_dset     = org_h5.create_dataset('output_{}'.format(highest_res), shape=(num_examples, highest_res, highest_res, 3), dtype=np.uint8)
        org_emb_dset = org_h5.create_dataset('embedding', shape=(num_examples, 1024), dtype=np.float)
    else:
        org_dset = None
        org_emb_dset = None

    with h5py.File(save_h5,'w') as h5file:
        
        start_count = 0
        data_count = {}
        dset = {}
        gen_samples = []
        img_samples = []
        vis_samples = {}
        tmp_samples = {}
        init_flag = True
        to_img_dict = functools.partial(to_img_dict_, super512=args.finest_size == 512)

        while True:
            if start_count >= num_examples:
                break
            test_images, test_embeddings_list, test_captions, saveIDs, classIDs = test_sampler()
            
            this_batch_size =  test_images.shape[0]
            chosen_captions = []
            for this_caption_list in test_captions:
                chosen_captions.append(this_caption_list[0])

            all_choosen_caption.extend(chosen_captions)    
            if org_dset is not None:
                org_dset[start_count:start_count+this_batch_size] = ((test_images + 1) * 127.5 ).astype(np.uint8)
                org_emb_dset[start_count:start_count+this_batch_size] = test_embeddings_list[0]

            start_count += this_batch_size
            
            # test_embeddings_list is a list of (B,emb_dim)
            
            for t in range(args.test_sample_num):
                
                B = len(test_embeddings_list)
                ridx = random.randint(0, B-1)
                testing_z.data.normal_(0, 1)

                this_test_embeddings_np = test_embeddings_list[ridx]
                this_test_embeddings    = to_device(this_test_embeddings_np)
                test_outputs, _         = to_img_dict(netG(this_test_embeddings, testing_z[0:this_batch_size]))
                
                if  t == 0: 
                    if init_flag is True:
                        dset['saveIDs']   = h5file.create_dataset('saveIDs', shape=(total_number,), dtype=np.int64)
                        dset['classIDs']  = h5file.create_dataset('classIDs', shape=(total_number,), dtype=np.int64)
                        dset['embedding'] = h5file.create_dataset('embedding', shape=(total_number, 1024), dtype=np.float)
                        for k in test_outputs.keys():
                            vis_samples[k] = [None for i in range(args.test_sample_num + 1)] # +1 to fill real image
                            img_shape = test_outputs[k].size()[2::]
    
                            dset[k] = h5file.create_dataset(k, shape=(total_number,)+ img_shape + (3,), dtype=np.uint8)
                            data_count[k] = 0
                    init_flag = False    
                
                for typ, img_val in test_outputs.items():
                    cpu_data = img_val.cpu().data.numpy()
                    row, col = cpu_data.shape[2],cpu_data.shape[3] 
                    if t==0:
                        #print("test_images shape: ", test_images.shape, np.mean(test_images))
                        this_reshape = imresize_shape(test_images,  (row, col) ) 
                        this_reshape = this_reshape * (2. / 255) - 1.
                        
                        this_reshape = this_reshape.transpose(0, 3, 1, 2)
                        vis_samples[typ][0] = this_reshape
                        #print("this_reshape shape: ", this_reshape.shape, np.mean(this_reshape))

                    vis_samples[typ][t+1]   = cpu_data
                    bs = cpu_data.shape[0]
                    
                    start = data_count[typ]
                    this_sample = ((cpu_data + 1) * 127.5 ).astype(np.uint8)
                    this_sample = this_sample.transpose(0, 2,3,1)

                    dset[typ][start: start + bs] = this_sample
                    dset['saveIDs'][start: start + bs] = saveIDs
                    dset['classIDs'][start: start + bs] = classIDs
                    #print(start, start+bs, dset['embedding'].shape, this_test_embeddings_np.shape)
                    dset['embedding'][start: start + bs] = this_test_embeddings_np #np.tile(this_test_embeddings_np, (bs,1))
                    data_count[typ] = start + bs
            
            if args.save_visual_results:
                save_super_images(vis_samples, chosen_captions, this_batch_size, save_folder, saveIDs, classIDs)

            print('saved files [sample {}/{}]: '.format(start_count, num_examples), data_count)  
            
        caption_array = np.array(all_choosen_caption, dtype=object)
        string_dt = h5py.special_dtype(vlen=str)
        h5file.create_dataset("captions", data=caption_array, dtype=string_dt)
        if org_dset is not None:
            org_h5.close() 

#-----------------------------------------------------------------------------------------------#
#  drawCaption and save_super_images is modified from https://github.com/hanzhanggit/StackGAN   #
#-----------------------------------------------------------------------------------------------#
def drawCaption(img, caption, level=['output 64', 'output 128', 'output 256']):
    img_txt = Image.fromarray(img)
    # get a font
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)

    # draw text, half opacity
    for idx, this_level in enumerate(level):
        d.text((10, 256 + idx * 256), this_level, font=fnt, fill=(255, 255, 255, 255))

    idx = caption.find(' ', 60)
    if idx == -1:
        d.text((256, 10), caption, font=fnt, fill=(255, 255, 255, 255))
    else:
        cap1 = caption[:idx]
        cap2 = caption[idx+1:]
        d.text((256, 10), cap1, font=fnt, fill=(255, 255, 255, 255))
        d.text((256, 60), cap2, font=fnt, fill=(255, 255, 255, 255))

    return img_txt

def save_super_images(vis_samples, captions_batch, batch_size, save_folder, saveIDs, classIDs, max_sample_num=8, save_single_img=True):
    save_folder_caption = os.path.join(save_folder, 'with_captions')
    save_folder_images  = os.path.join(save_folder, 'images')
    
    dst_shape = (0,0)
    all_row = []
    level = []
    for typ, img_list in vis_samples.items():
        this_shape = img_list[0].shape[2::] # bs, 3, row, col
        if this_shape[0] > dst_shape[0]:
            dst_shape = this_shape
        level.append(typ)

    valid_caption = []
    valid_IDS = []
    valid_classIDS = []
    for j in range(batch_size):
        if not re.search('[a-zA-Z]+', captions_batch[j]):
            print("Not valid caption? :",  captions_batch[j])
            continue
        else:  
            valid_caption.append(captions_batch[j])
            valid_IDS.append(saveIDs[j])
            valid_classIDS.append(classIDs[j])

    for typ, img_list in vis_samples.items(): 
        img_tensor = np.stack(img_list, 1) # N * T * 3 *row*col
        img_tensor = img_tensor.transpose(0,1,3,4,2)
        img_tensor = (img_tensor + 1.0) * 127.5
        img_tensor = img_tensor.astype(np.uint8)

        this_img_list = []

        batch_size  = img_tensor.shape[0]
        #imshow(img_tensor[0,0])
        batch_all = []
        for bidx in range(batch_size):
            if save_single_img:
                this_folder_id = os.path.join(save_folder_images, '{}_{}'.format(valid_classIDS[bidx], valid_IDS[bidx]))
                mkdirs([this_folder_id])

            if not re.search('[a-zA-Z]+', captions_batch[j]):
                continue
            padding = np.zeros(dst_shape + (3,), dtype=np.uint8)
            this_row = [padding]
            # First row with up to 8 samples
            for tidx in range(img_tensor.shape[1]):
                this_img  = img_tensor[bidx][tidx]
                
                re_sample = imresize_shape(this_img, dst_shape)
                if tidx <= max_sample_num:
                    this_row.append(re_sample)  
                #img_rgb = ( (re_sample + 1.0) * 127.5 ).astype(np.uint8)
                #print("img_rgb shape: ", img_rgb.shape)

                ## TODO to save space, do we not save single image here. You can do that if you want
                if save_single_img:
                    scipy.misc.imsave(os.path.join(this_folder_id, '{}_copy_{}.jpg'.format(typ, tidx)),  re_sample)
                
            this_row = np.concatenate(this_row, axis=1) # row, col*T, 3
            batch_all.append(this_row)
        batch_all = np.stack(batch_all, 0) # bs*row*colT*3 
        all_row.append(batch_all)

    all_row = np.stack(all_row, 0) # n_type * bs * shape    
    
    batch_size = len(valid_IDS) 
    
    mkdirs([save_folder_caption, save_folder_images])
    for idx in range(batch_size):
        this_select = all_row[:, idx] # ntype*row*col
        
        ntype, row, col, chn = this_select.shape
        superimage = np.reshape(this_select, (-1, col, chn) )  # big_row, col, 3

        top_padding = np.zeros((128, superimage.shape[1], 3))
        superimage =\
            np.concatenate([top_padding, superimage], axis=0)
            
        save_path = os.path.join(save_folder_caption, '{}_{}.png'.format(valid_classIDS[idx], valid_IDS[idx]) )    
        superimage = drawCaption(np.uint8(superimage), valid_caption[idx], level)
        scipy.misc.imsave(save_path, superimage)


