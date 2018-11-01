from __future__ import print_function
'''
sentence processing code modified from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
'''
import sys
sys.path.insert(0,'coco-master/PythonAPI/')
from pycocotools.coco import COCO
import numpy as np
from random import shuffle
from orderedset import OrderedSet

from collections import defaultdict, OrderedDict
from PIL import Image
import os as os
# from log_utils import print
import tables, json
import pandas as pd
import pickle
from pprint import pprint
# from data_helpers import *
# from w2v import train_word2vec
import torchfile as torchf

nrows, ncols, nch = 256, 256, 3

def load_data(data_type,
                sequence_length=128,
                embedding_dim=128, context=10,
                min_word_count=1, nch_sent=5):
    data_dir = '.'
    annFile = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
    capsFile = '{}/annotations/captions_{}.json'.format(data_dir, data_type)
    hdf5_path = '{}/dataset_{}.h5'.format(data_dir, data_type) # path to save the hdf5 file
    voc_pick_File = '{}/mr.p'.format(data_dir)
    embedded_caption_path = '{}/embedded_caption_{}.h5'.format(data_dir, data_type)
    #w2v_file = './coco_official/GoogleNews-vectors-negative300.bin'
    
    if not os.path.exists(hdf5_path):
        # initialize COCO api for instance annotations
        coco = COCO(annFile)
        coco_caps = COCO(capsFile)

        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        sup_nms = OrderedSet([cat['supercategory'] for cat in cats])

        # get all images containing given categories
        catIds = coco.getCatIds(catNms=nms)
        imgIds = coco.getImgIds()

        data_shape = (len(imgIds), nrows, ncols, nch)
        fine_label_shape = (len(imgIds), 80)
        coarse_label_shape = (len(imgIds), )

        # images = np.ndarray(shape=data_shape, dtype=np.uint8)
        # fine_labels = np.ndarray(shape=fine_label_shape, dtype=np.uint8)
        # coarse_labels = np.ndarray(shape=coarse_label_shape, dtype=np.uint8)
        # print("Shape till now: ", images.shape)

        idx  = 0 #index for images
        sentences = []
        tmp_captions = OrderedDict()
        
        for imgId in imgIds[:]:
            # label = [0] * 80
            # for i, catId in enumerate(catIds):
            #     ids = coco.getImgIds(catIds=catId) # get images of this particular category
            #     if imgId in ids:
            #         label[i] = 1
            #         cat = coco.loadCats(catId)[0]
            #         # coarse_labels[idx] = sup_nms.index(cat['supercategory'])

            # fine_labels[idx] = label
            img = coco.loadImgs(imgId)[0]
            # path = ('%s/%s/%s'%(data_dir, data_type, img['file_name']))
            # image = Image.open(path).convert('RGB')
            # image = image.resize((nrows, ncols), Image.ANTIALIAS)
            # images[idx] = np.asarray(image)
            # load and display caption annotations
            annIds = coco_caps.getAnnIds(imgIds=img['id'])
            anns = coco_caps.loadAnns(annIds)
            tmp_caption = []
            for i, ann in enumerate(anns):
                tmp_caption.append(ann['caption'])
            
            tmp_captions[img['file_name']] = tmp_caption
            # import pdb; pdb.set_trace()

            idx += 1
            if idx % 100 == 0:
                print ('load image {} with {} captions'.format(img['file_name'], len(tmp_caption)))
                print ('{} images collectd ... '.format(idx))
            
            with open('/home/zizhaozhang/work/LaplacianGan/Data/coco/val2014_ex_t7/caption_txt/{}.txt'.format(img['file_name']),'w') as f:
                for c in tmp_caption:
                    f.write('{}\n'.format(c))

        # save everything
        path = '/home/zizhaozhang/work/LaplacianGan/Data/coco/' + data_type[:-4]

        pickle.dump(tmp_captions, open(os.path.join(path, 'captions.pickle'), 'wb'))
        #pickle.dump(images, open(os.path.join(path, data_type+'_256images.pickle'), 'wb'))
        # json.dump(tmp_captions, open(os.path.join(path, data_type+'_captions.json'), 'wb'))
        # torchf.save(os.path.join(path, data_type+'_captions.t7'),tmp_captions)
        

    #     if not os.path.exists(voc_pick_File):
    #         print("loading data...")
    #         """
    #         Loads and preprocessed data for the MR dataset.
    #         Returns input vectors, labels, vocabulary, and inverse vocabulary.
    #         """
    #         # Load and preprocess data
    #         sentences_padded, sequence_length = pad_sentences(sentences, max_length=sequence_length)
    #         vocab, vocab_inv = build_vocab(sentences_padded)
    #         data_x = build_input_data(sentences_padded, vocab)
    #         vocab_inv = {key: value for key, value in enumerate(vocab_inv)}

    #         print("data loaded!")
    #         print("number of sentences: " + str(len(data_x)))
    #         print("vocab size: " + str(len(vocab)))
    #         print("max sentence length: " + str(sequence_length))
    #         print("loading word2vec vectors...")
    #         embedding_weights = train_word2vec(data_x,
    #                             vocab_inv,
    #                             num_features=embedding_dim,
    #                             min_word_count=min_word_count,
    #                             context=context)
    #         cPickle.dump([vocab, vocab_inv, sequence_length, embedding_weights], open(voc_pick_File, "wb"))
    #         print("dataset created!")
    #     else:
    #         x = cPickle.load(open(voc_pick_File, "rb"))
    #         vocab, vocab_inv, sequence_length, embedding_weights = x[0], x[1], x[2], x[3]
    #         print("data loaded!")

    #     caption_shape = (len(imgIds), nch_sent, sequence_length)
    #     captions = np.ndarray(shape=caption_shape, dtype=np.int)
    #     for i, tmp in enumerate(tmp_captions):
    #         tmp_pad, _ = pad_sentences(tmp, max_length=sequence_length)
    #         tmp_pad = build_input_data(tmp_pad, vocab) # map word to index
    #         captions[i] = tmp_pad
    #         print("shape of captions: ", captions[i].shape)

    #     print("Saving the dataset...")
    #     # open a hdf5 file and create earrays
    #     with tables.open_file(hdf5_path, mode='w') as hdf5_file:
    #         # create the label arrays and copy the labels data in them
    #         hdf5_file.create_array(hdf5_file.root, 'images', images)
    #         hdf5_file.create_array(hdf5_file.root, 'fine_labels', fine_labels)
    #         hdf5_file.create_array(hdf5_file.root, 'coarse_labels', coarse_labels)
    #         hdf5_file.create_array(hdf5_file.root, 'captions', captions)
    # else:
    #     x = cPickle.load(open(voc_pick_File, "rb"))
    #     _, vocab_inv, _, embedding_weights = x[0], x[1], x[2], x[3]
    #     print("data loaded!")

    #     print("Loading the dataset...")
    #     with tables.open_file(hdf5_path, mode='r') as hdf5_file:
    #         images = hdf5_file.root.images[:]
    #         fine_labels = hdf5_file.root.fine_labels[:]
    #         coarse_labels = hdf5_file.root.coarse_labels[:]
    #         captions = hdf5_file.root.captions[:]

    # embedded_caption = []

    # if os.path.exists(embedded_caption_path):
    #     with tables.open_file(embedded_caption_path, mode='r') as hdf5_file:
    #         embedded_caption = hdf5_file.root.embedded_caption[:]

    # print("Loaded embedded caption of shape {}\n".format(np.shape(embedded_caption)))

    # return images, fine_labels, coarse_labels, captions, embedding_weights, len(vocab_inv), embedded_caption

def generate_ranking_info(batch_label):
    '''
    (fine_labels, coarse_labels) = y_train[i][0], y_train[i][1]
    '''
    n = batch_label.shape[0]
    S = -np.ones((n, n))
    for iter in range(n):
        for jter in range(n):
            tepV = np.matmul(batch_label[iter, :], np.transpose(batch_label[jter, :]))
            if tepV != 0:
                S[iter, jter] = tepV
    return S

if __name__ == '__main__':
    load_data('train2014')
    load_data('val2014')