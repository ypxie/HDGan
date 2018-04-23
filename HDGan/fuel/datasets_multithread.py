#-------------------------------------------------------------------------#
# dataloader for birds and flowers is modified from https://github.com/hanzhanggit/StackGAN
# don't set batch size 1
#-------------------------------------------------------------------------#
import numpy as np
import pickle
import random
from collections import OrderedDict
import sys, os
import scipy.misc as misc
import torch.utils.data
from functools import partial


def resize_images(tensor, shape):
    out = []
    for k in range(tensor.shape[0]):
        tmp = misc.imresize(tensor[k], shape)
        out.append(tmp[np.newaxis,:,:,:])
    return np.concatenate(out, axis=0)

def img_loader_func(img_names, imgpath=None, img_size=256):
    res = []

    for i_n in img_names:
        img = misc.imread(os.path.join(imgpath, i_n))
        img = misc.imresize(img, (img_size, img_size))
        if len(img.shape) != 3:
            # happen to be a gray image
            img = np.tile(img[:,:,np.newaxis], [1,1,3])

        res.append(img[np.newaxis,:,:,:])
    res = np.concatenate(res, axis=0)
    
    return res

class Dataset(object):
    def __init__(self, workdir, img_size, batch_size, n_embed, mode='train'):
        
        if img_size in [256, 512]:
            self.output_res = [64, 128, 256]
            if img_size == 512: self.output_res += [512]
        elif img_size in [64]:
            self.output_res = [64]

        self.embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        self.image_shape = [img_size, img_size, 3]

        self.batch_size = batch_size
        self.n_embed = n_embed

        self.imsize = img_size
        self.workdir = workdir
        self.train_mode = mode == 'train'
        self.get_data(os.path.join(self.workdir, mode))

        print('-> init data loader ', mode)
        print('\t {} samples'.format(self._num_examples))
        print('\t {} output resolutions'.format(self.output_res))
        
    def get_data(self, data_dir):
        
        data_root = os.path.split(data_dir)[0]
        if self.train_mode:
            img_path = os.path.join(data_root, 'coco_official', 'train2014')
        else:
            img_path = os.path.join(data_root,'coco_official', 'val2014')
        
        self.images = partial(img_loader_func, imgpath=img_path, img_size=self.imsize)
        
        with open(data_dir + self.embedding_filename, 'rb') as f:
            if sys.version_info.major > 2:
                embeddings = pickle.load(f,  encoding="bytes")
            else:
                embeddings = pickle.load(f)

            self.embeddings = np.array(embeddings)
            self.embedding_shape = [self.embeddings.shape[-1]]
            # print('embeddings: ', self.embeddings.shape)

        with open(data_dir + '/filenames.pickle', 'rb') as f:
            self.filenames = pickle.load(f)
            # print('list_filenames: ', len(self.filenames))

        with open(data_dir + '/captions.pickle', 'rb') as f:
            self.captions = pickle.load(f)
            # print('read {} captions '.format(len(self.captions)))
        
        self._num_examples = len(self.filenames)
        
    def readCaptions(self, filenames):
        cap = self.captions[filenames]
        return cap

    def transform(self, images):
        
        transformed_images = np.zeros([images.shape[0], self.imsize, self.imsize, 3])
        ori_size = images.shape[1]
        for i in range(images.shape[0]):
            if self.train_mode:
                h1 = int( np.floor((ori_size - self.imsize) * np.random.random()) )
                w1 = int( np.floor((ori_size - self.imsize) * np.random.random()) )
            else:
                h1 = int(np.floor((ori_size - self.imsize) * 0.5))
                w1 = int(np.floor((ori_size - self.imsize) * 0.5))
                
            cropped_image = images[i][w1:w1 + self.imsize, h1:h1 + self.imsize, :]

            if random.random() > 0.5:
                transformed_images[i] = np.fliplr(cropped_image)
            else:
                transformed_images[i] = cropped_image

        return transformed_images

    def sample_embeddings(self, embeddings, filenames, sample_num):
        if len(embeddings.shape) == 2 or embeddings.shape[1] == 1:
            return np.squeeze(embeddings)
        else:
            batch_size, embedding_num, _ = embeddings.shape
            # Take every sample_num captions to compute the mean vector
            sampled_embeddings = []
            sampled_captions = []
            for i in range(batch_size):
                randix = np.random.choice(embedding_num,
                                          sample_num, replace=False)
                if sample_num == 1:
                    randix = int(randix)
                    captions = self.readCaptions(filenames[i])
                    sampled_captions.append(captions[randix])
                    sampled_embeddings.append(embeddings[i, randix, :])
                else:
                    e_sample = embeddings[i, randix, :]
                    e_mean = np.mean(e_sample, axis=0)
                    captions = self.readCaptions(filenames[i])
                    sampled_captions.append(captions[randix[0]])

                    sampled_embeddings.append(e_mean)
            sampled_embeddings_array = np.array(sampled_embeddings)
            
            return np.squeeze(sampled_embeddings_array), sampled_captions

    def __getitem__(self, index):
        """Return the next `batch_size` examples from this data set."""

        current_ids = [index] # only take one
        fake_ids = np.random.randint(self._num_examples, size=len(current_ids))
        
        images_dict = OrderedDict()
        wrongs_dict = OrderedDict()

        filenames = [self.filenames[i].decode() for i in current_ids]
        fake_filenames = [self.filenames[i].decode() for i in fake_ids]

        sampled_images = self.images(filenames)
        sampled_wrong_images = self.images(fake_filenames)
        sampled_images = sampled_images.astype(np.float32)
        sampled_wrong_images = sampled_wrong_images.astype(np.float32)
        sampled_images = self.transform(sampled_images)
        sampled_wrong_images = self.transform(sampled_wrong_images)
        images_dict = {}
        wrongs_dict = {}
        
        for size in self.output_res:
            tmp = resize_images(sampled_images, shape=[size, size]).transpose((0,3,1,2))
            tmp = tmp * (2. / 255) - 1.
            tmp = np.squeeze(tmp, 0)
            images_dict['output_{}'.format(size)] = tmp.astype(np.float32)
            tmp = resize_images(sampled_wrong_images, shape=[size, size]).transpose((0,3,1,2))
            tmp = tmp * (2. / 255) - 1.
            tmp = np.squeeze(tmp, 0)
            wrongs_dict['output_{}'.format(size)] = tmp.astype(np.float32)

        ret_list = [images_dict, wrongs_dict]

        sampled_embeddings, sampled_captions = \
            self.sample_embeddings(self.embeddings[current_ids],
                                    filenames, self.n_embed)
        ret_list.append(sampled_embeddings)
        ret_list.append(sampled_captions)

        ret_list.append(filenames)

        return ret_list

    def next_batch_test(self, max_captions=1):
        """Return the next `batch_size` examples from this data set."""
        batch_size = self.batch_size
        
        start = self._text_index
        if (start + batch_size) > self._num_examples:
            end = self._num_examples
            self._text_index = 0
        else:
            end = start + batch_size
        self._text_index += batch_size

        sampled_images = self.images[start:end]
        sampled_images = sampled_images.astype(np.float32)
        sampled_images = self.transform(sampled_images)
        # from [0, 255] to [-1.0, 1.0]
        sampled_images = sampled_images * (2. / 255) - 1.
        
        sampled_embeddings = self.embeddings[start:end]
        _, embedding_num, _ = sampled_embeddings.shape
        sampled_embeddings_batchs = []
        
        sampled_captions = []
        sampled_filenames = self.filenames[start:end]
        for i in range(len(sampled_filenames)):
            captions = self.readCaptions(sampled_filenames[i])
            sampled_captions.append(captions)

        for i in range(np.minimum(max_captions, embedding_num)):
            batch = sampled_embeddings[:, i, :]
            sampled_embeddings_batchs.append(batch)

        return [sampled_images, sampled_embeddings_batchs, sampled_captions]

    def __len__(self):
        return self._num_examples

class COCODataset():

    def __init__(self, data_dir, img_size, batch_size, num_embed, mode='train', threads=0, drop_last=True):
        print ('create multithread loader with {} threads ...'.format(threads))

        self.dataset = Dataset(data_dir, img_size=img_size, batch_size=batch_size, n_embed=num_embed, mode=mode)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=mode=='train',
            num_workers=threads,
            drop_last=drop_last)
            
        self._num_examples = len(self.dataset)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

