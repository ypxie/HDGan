import numpy as np
import pickle
import random
from collections import OrderedDict
import sys, os
import scipy.misc as misc
from ..proj_utils.local_utils import imresize_shape
import torch

from functools import partial

IMG_DIM = 304 # use for augmentation
def resize_images(tensor, shape):
    out = []
    for k in range(tensor.shape[0]):
        tmp = misc.imresize(tensor[k], shape)
        out.append(tmp[np.newaxis,:,:,:])
    return np.concatenate(out, axis=0).transpose((0,3,1,2))

def img_loader_func(img_names, imgpath=None):
    res = []

    for i_n in img_names:
        #print (imgpath, i_n)
        img = misc.imread(os.path.join(imgpath, i_n))
        img = misc.imresize(img, (IMG_DIM, IMG_DIM))
        if len(img.shape) != 3:
            # happen to be a gray image
            img = np.tile(img[:,:,np.newaxis], [1,1,3])

        res.append(img[np.newaxis,:,:,:])
    res = np.concatenate(res, axis=0)
    
    return res

# bugs if you use batch size 1
class Dataset(object):
    def __init__(self, images, imsize, embeddings=None,
                 filenames=None, workdir=None,
                 labels=None, aug_flag=True,
                 class_id=None, class_range=None, side_list=[64, 128], captions=None):
        self._images = images
        self._embeddings = embeddings
        self._filenames = filenames
        self.workdir = workdir
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = len(filenames)
        self._saveIDs = self.saveIDs()

        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._aug_flag = aug_flag
        self._class_id = np.array(class_id)
        self._class_range = class_range
        self._imsize = imsize
        self._perm = None
        self.end_of_data = False

        self.captions = captions

    @property
    def images(self):
        return self._images

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def filenames(self):
        return self._filenames

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def saveIDs(self):
        self._saveIDs = np.arange(self._num_examples)
        #np.random.shuffle(self._saveIDs) why do we need to shuffle ????
        return self._saveIDs

    def readCaptions(self, filenames):
        #import pdb; pdb.set_trace()
        
        cap = self.captions[filenames]
        # print ('can not find {} in captions'.format(filenames))
        return cap

    def transform(self, images):
        
        if self._aug_flag:
            transformed_images =\
                np.zeros([images.shape[0], self._imsize, self._imsize, 3])
            ori_size = images.shape[1]
            for i in range(images.shape[0]):
                h1 = int( np.floor((ori_size - self._imsize) * np.random.random()) )
                w1 = int( np.floor((ori_size - self._imsize) * np.random.random()) )

                cropped_image =\
                    images[i][w1: w1 + self._imsize, h1: h1 + self._imsize, :]
                if random.random() > 0.5:
                    transformed_images[i] = np.fliplr(cropped_image)
                else:
                    transformed_images[i] = cropped_image
            return transformed_images
        else:
            return images

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
                    # if has many load the first one
                    captions = self.readCaptions(filenames[i])
                    sampled_captions.append(captions[randix[0]])
                    
                    sampled_embeddings.append(e_mean)
            sampled_embeddings_array = np.array(sampled_embeddings)
            return np.squeeze(sampled_embeddings_array), sampled_captions

    def next_batch(self, index, window):
        """Return the next `batch_size` examples from this data set."""

        # start = self._index_in_epoch
        # self._index_in_epoch += batch_size
        # print (self._index_in_epoch)
        # if self._index_in_epoch > self._num_examples:
        #     # Finished epoch
        #     self._epochs_completed += 1
        #     # Shuffle the data
        #     self._perm = np.arange(self._num_examples)
        #     np.random.shuffle(self._perm)

        #     # Start next epoch
        #     start = 0
        #     self._index_in_epoch = batch_size
        #     assert batch_size <= self._num_examples
        #     #print (self._index_in_epoch,  self._num_examples)
        #     #print ('go to next round')

        # end = self._index_in_epoch

        # current_ids = self._perm[start:end]
        current_ids = [index] # only take one
        
        fake_ids = np.random.randint(self._num_examples, size=len(current_ids))

        # collision_flag =\
        #     (self._class_id[current_ids] == self._class_id[fake_ids])
        # fake_ids[collision_flag] =\
        #     (fake_ids[collision_flag] +
        #      np.random.randint(100, 200)) % self._num_examples
        
        images_dict = OrderedDict()
        wrongs_dict = OrderedDict()
        
        filenames = [self._filenames[i].decode() for i in current_ids]
        fake_filenames = [self._filenames[i].decode() for i in fake_ids]
        # import pdb; pdb.set_trace()
        sampled_images = self._images(filenames)
        sampled_wrong_images = self._images(fake_filenames)
        
        sampled_images = sampled_images
        sampled_wrong_images = sampled_wrong_images
        sampled_images = self.transform(sampled_images)
        sampled_wrong_images = self.transform(sampled_wrong_images)

        sampled_images = sampled_images.astype(np.float32)
        sampled_wrong_images = sampled_wrong_images.astype(np.float32)

        images_dict = {}
        wrongs_dict = {}
        for size in [64, 128, 256]:
            tmp = resize_images(sampled_images, shape=[size, size])
            tmp = tmp * (2. / 255) - 1.
            tmp = np.squeeze(tmp, 0) # squeee is to remove the extra dimensions since we use pytorch multithread loader
            images_dict['output_{}'.format(size)] = tmp.astype(np.float32)
            
            tmp = resize_images(sampled_wrong_images, shape=[size, size])
            tmp = tmp * (2. / 255) - 1.
            tmp = np.squeeze(tmp, 0)
            wrongs_dict['output_{}'.format(size)] = tmp.astype(np.float32)


        ret_list = [images_dict, wrongs_dict]

        if self._embeddings is not None:
            # class_id = [self._class_id[i] for i in current_ids]
            sampled_embeddings, sampled_captions = \
                self.sample_embeddings(self._embeddings[current_ids],
                                       filenames, window)
            ret_list.append(sampled_embeddings.astype(np.float32))
            ret_list.append(sampled_captions)
        else:
            ret_list.append([])
            ret_list.append([])

        # if self._labels is not None:
        #     ret_list.append(self._labels[current_ids])
        # else:
        #     ret_list.append([])

        ret_list.append(filenames)

        return ret_list


    def next_batch_test(self, batch_size, start, max_captions):
        """Return the next `batch_size` examples from this data set."""
        
        if (start + batch_size) > self._num_examples:
            end = self._num_examples
        else:
            end = start + batch_size
        
        sampled_filenames = [self._filenames[i].decode() for i in range(start, end)] 
        sampled_images = self._images(sampled_filenames)
        
        #sampled_images = sampled_images
        sampled_images = resize_images(sampled_images, shape=[256, 256])
        # from [0, 255] to [-1.0, 1.0]
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_images = self.transform(sampled_images)
        sampled_images = sampled_images.astype(np.float32)
        sampled_images = sampled_images.transpose(0, 2,3,1)
        sampled_embeddings = self._embeddings[start:end]
        _, embedding_num, _ = sampled_embeddings.shape
        sampled_embeddings_batchs = []
        
        sampled_captions = []
        
        #print(type(self._class_id), start, end, len(self._class_id))
        #sampled_class_id  = self._class_id[start:end]
        for i in range(len(sampled_filenames)):
            captions = self.readCaptions(sampled_filenames[i])
            # print(captions)
            sampled_captions.append(captions)

        for i in range(np.minimum(max_captions, embedding_num)):
            batch = sampled_embeddings[:, i, :]
            sampled_embeddings_batchs.append(np.squeeze(batch))

        return [sampled_images, sampled_embeddings_batchs,
                self._saveIDs[start:end], sampled_captions, sampled_filenames]


class TextDataset(object):
    def __init__(self, hr_lr_ratio=4):
        lr_imsize = 64
        self.hr_lr_ratio = hr_lr_ratio

        # if self.hr_lr_ratio == 1:
        #     self.image_filename = '/76images.pickle'
        # elif self.hr_lr_ratio == 4:
        #     self.image_filename = '/304images.pickle'

        self.image_shape = [lr_imsize * self.hr_lr_ratio,
                            lr_imsize * self.hr_lr_ratio, 3]
        self.image_dim = self.image_shape[0] * self.image_shape[1] * 3
        self.embedding_shape = None
        self.train = None
        self.test = None
        # self.workdir = workdir
        self.embedding_filename = '/char-CNN-RNN-embeddings.pickle'


    def get_data(self, pickle_path, aug_flag=True, data_dir=None):
        # with open(pickle_path + self.image_filename, 'rb') as f:
        #     images = pickle.load(f)
        #     images = np.array(images)
        if 'train' in pickle_path:
            IMG_PATH = os.path.join(data_dir,'coco_official', 'train2014')
        elif 'val' in pickle_path:
            IMG_PATH = os.path.join(data_dir,'coco_official', 'val2014')
        
        print ('read data from {}'.format(IMG_PATH))
        img_load = partial(img_loader_func, imgpath=IMG_PATH)

        with open(pickle_path + self.embedding_filename, 'rb') as f:
            if sys.version_info.major > 2:
                embeddings = pickle.load(f, encoding="bytes")
            else:
                embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
            self.embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)

        with open(pickle_path + '/filenames.pickle', 'rb') as f:
            list_filenames = pickle.load(f)
            print('list_filenames: ', len(list_filenames), list_filenames[0])
            
        with open(pickle_path + '/captions.pickle', 'rb') as f:
            captions = pickle.load(f)
            print ('read {} captions '.format(len(captions)))
        return Dataset(img_load, self.image_shape[0], embeddings,
                       list_filenames, None, None,
                       aug_flag, captions=captions)


class WrapperLoader():
    def __init__(self, pickle_path, num_embed, test_mode, aug_flag,data_dir):
        
        self.dataset = TextDataset().get_data(pickle_path, aug_flag=aug_flag, data_dir=data_dir)
        self.num_embed = num_embed
        self.num_examples = self.dataset.num_examples
        self.num_samples  = self.num_examples
        self.test_mode = test_mode

    def __getitem__(self, index):
        if self.test_mode:
            # data = self.dataset.next_batch_test(1, index, self.num_embed)
            data = self.dataset.next_batch(index, self.num_embed)
        else:
            data = self.dataset.next_batch(index, self.num_embed)

        return data

    def __len__(self):
        return self.num_examples

class MultiThreadLoader():

    def __init__(self, pickle_path, batch_size, num_embed, threads=0, test_mode=False, aug_flag=True,data_dir=None,drop_last=True):
        print ('create multithread loader with {} threads ...'.format(threads))

        self.dataset = WrapperLoader(pickle_path, num_embed, test_mode, aug_flag=aug_flag, data_dir=data_dir)
        import torch.utils.data
        print ('length of dataset', len(self.dataset))
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=threads,
            drop_last = drop_last)
        self.dataloader.num_examples = self.dataset.num_examples

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)

