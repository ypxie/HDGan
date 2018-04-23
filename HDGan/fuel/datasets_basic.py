import numpy as np
import pickle
import random
from collections import OrderedDict
import sys, os
import scipy.misc as misc

#-------------------------------------------------------------------------#
# dataloader for birds and flowers is modified from https://github.com/hanzhanggit/StackGAN
# don't set batch size 1
#-------------------------------------------------------------------------#

def resize_images(tensor, shape):
    out = []
    for k in range(tensor.shape[0]):
        tmp = misc.imresize(tensor[k], shape)
        out.append(tmp[np.newaxis,:,:,:])
    return np.concatenate(out, axis=0)

class Dataset(object):
    def __init__(self, workdir, img_size, batch_size, n_embed, mode='train'):
        
       
        if img_size in [256, 512]:
            self.image_filename = '/304images.pickle'
            self.output_res = [64, 128, 256]
            if img_size == 512: self.output_res += [512]
        elif img_size in [64]:
            self.image_filename = '/76images.pickle'
            self.output_res = [64]
        self.embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        self.image_shape = [img_size, img_size, 3]
        # self.image_dim = self.image_shape[0] * self.image_shape[1] * 3
        # self.embedding_shape = None
        # self.train = None
        # self.test = None
        self.batch_size = batch_size
        self.n_embed = n_embed

        self.imsize = img_size
        self.workdir = workdir
        self.train_mode = mode == 'train'
        self.get_data(os.path.join(self.workdir, mode))

        # set up sampler
        self._index_in_epoch = 0
        self._text_index = 0
        self._perm = np.arange(self._num_examples)
        np.random.shuffle(self._perm)
        self._epochs_completed = -1
        self.saveIDs = np.arange(self._num_examples)

        print('-> init data loader ', mode)
        print('\t {} samples'.format(self._num_examples))
        print('\t {} output resolutions'.format(self.output_res))
        
    def get_data(self, pickle_path):
        with open(pickle_path + self.image_filename, 'rb') as f:
            images = pickle.load(f)
            self.images = np.array(images)

        with open(pickle_path + self.embedding_filename, 'rb') as f:
            if sys.version_info.major > 2:
                embeddings = pickle.load(f,  encoding="bytes")
            else:
                embeddings = pickle.load(f)

            self.embeddings = np.array(embeddings)
            self.embedding_shape = [self.embeddings.shape[-1]]
            # print('embeddings: ', self.embeddings.shape)

        with open(pickle_path + '/filenames.pickle', 'rb') as f:
            self.filenames = pickle.load(f)
            # print('list_filenames: ', len(self.filenames))

        with open(pickle_path + '/class_info.pickle', 'rb') as f:
            if sys.version_info.major > 2:
                class_id = pickle.load(f, encoding="bytes")
            else:
                class_id = pickle.load(f)
            self.class_id = np.array(class_id)

        self._num_examples = len(self.images)
        
    def readCaptions(self, filenames, class_id):
        name = filenames
        if name.find('jpg/') != -1:  # flowers dataset
            class_name = 'class_{0:05d}/'.format(class_id)
            name = name.replace('jpg/', class_name)
        cap_path = '{}/text_c10/{}.txt'.format(self.workdir, name)
        
        with open(cap_path, "r") as f:
            captions = f.read().split('\n')
        captions = [cap for cap in captions if len(cap) > 0]
        return captions

    def transform(self, images):
        
        transformed_images = np.zeros([images.shape[0], self.imsize, self.imsize, 3])
        ori_size = images.shape[1]
        for i in range(images.shape[0]):
            if self.train_mode:
                h1 = int( np.floor((ori_size - self.imsize) * np.random.random()) )
                w1 = int( np.floor((ori_size - self.imsize) * np.random.random()) )
            else:
                h1 = int(np.floor((ori_size - self.imsize) * 0.5))
                w1 = int( np.floor((ori_size - self.imsize) * 0.5))
                
            cropped_image =\
                images[i][w1: w1 + self.imsize, h1: h1 + self.imsize, :]
            if random.random() > 0.5:
                transformed_images[i] = np.fliplr(cropped_image)
            else:
                transformed_images[i] = cropped_image

        return transformed_images

    def sample_embeddings(self, embeddings, filenames, class_id, sample_num):
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
                    captions = self.readCaptions(filenames[i],
                                                 class_id[i])
                    sampled_captions.append(captions[randix])
                    sampled_embeddings.append(embeddings[i, randix, :])
                else:
                    e_sample = embeddings[i, randix, :]
                    e_mean = np.mean(e_sample, axis=0)
                    
                    sampled_embeddings.append(e_mean)
            sampled_embeddings_array = np.array(sampled_embeddings)
            
            return np.squeeze(sampled_embeddings_array), sampled_captions

    def __getitem__(self, idx):
        """Return the next `batch_size` examples from this data set."""

        n_embed = self.n_embed 
        
        start = self._index_in_epoch
        self._index_in_epoch += self.batch_size
        # shuffle
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            self._perm = np.arange(self._num_examples)
            np.random.shuffle(self._perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = self.batch_size
            assert self.batch_size <= self._num_examples
        end = self._index_in_epoch

        current_ids = self._perm[start:end]
        fake_ids = np.random.randint(self._num_examples, size=self.batch_size)

        collision_flag = (self.class_id[current_ids] == self.class_id[fake_ids])
        fake_ids[collision_flag] = (fake_ids[collision_flag] + np.random.randint(100, 200)) % self._num_examples
        
        images_dict = OrderedDict()
        wrongs_dict = OrderedDict()

        sampled_images = self.images[current_ids]
        sampled_wrong_images = self.images[fake_ids, :, :, :]
        sampled_images = sampled_images.astype(np.float32)
        sampled_wrong_images = sampled_wrong_images.astype(np.float32)
        sampled_images = self.transform(sampled_images)
        sampled_wrong_images = self.transform(sampled_wrong_images)
        images_dict = {}
        wrongs_dict = {}
        
        for size in self.output_res:
            tmp = resize_images(sampled_images, shape=[size, size]).transpose((0,3,1,2))
            tmp = tmp * (2. / 255) - 1.
            images_dict['output_{}'.format(size)] = tmp
            tmp = resize_images(sampled_wrong_images, shape=[size, size]).transpose((0,3,1,2))
            tmp = tmp * (2. / 255) - 1.
            wrongs_dict['output_{}'.format(size)] = tmp

        ret_list = [images_dict, wrongs_dict]

        filenames = [self.filenames[i] for i in current_ids]
        class_id = [self.class_id[i] for i in current_ids]
        sampled_embeddings, sampled_captions = \
            self.sample_embeddings(self.embeddings[current_ids],
                                    filenames, class_id, n_embed)
        ret_list.append(sampled_embeddings)
        ret_list.append(sampled_captions)
   
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
        sampled_class_id = self.class_id[start:end]
        for i in range(len(sampled_filenames)):
            captions = self.readCaptions(sampled_filenames[i],
                                         sampled_class_id[i])
            sampled_captions.append(captions)

        for i in range(np.minimum(max_captions, embedding_num)):
            batch = sampled_embeddings[:, i, :]
            sampled_embeddings_batchs.append(batch)

        return [sampled_images, sampled_embeddings_batchs, self.saveIDs[start:end],
                 self.class_id[start:end], sampled_captions]
