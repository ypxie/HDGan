import h5py, os

def load_data_from_h5(fullpath, h5_file):
    
    h5file = os.path.join(fullpath, h5_file)
    return_path = os.path.join(fullpath, h5_file[: - 3] + '_inception_score')
    print ('read h5 from {}'.format(h5file))
    assert(os.path.isfile(h5file))
    fh = h5py.File(h5file, 'r')
    
    keys = [a for a in fh.keys() if 'output' in a]
    ms_images = {a: [] for a in keys}

    sample_names = []
    classIDs = h5py.File(h5file)['classIDs']
    cls_idx = 0
    txt_idx = 0
    sample_idx = 0
    txt_idx_rest_iter = 10 if 'bird' in h5file else 26

    print ('evaluate scale ', ms_images.keys())
    for s, k in enumerate(ms_images.keys()):
            
        data = h5py.File(h5file)[k]
        images = []
        for i in range(data.shape[0]):
            img = data[i]
            # import pdb; pdb.set_trace()
            assert((img.shape[0] in [256, 128, 64, 512]) and img.shape[2] == 3)
            if not (img.min() >= 0 and img.max() <= 255 and img.mean() > 1):
                print ('WARNING {}, min {}, max {}, mean {}'.format(i, img.min(), img.max(), img.mean()))
                continue	
            assert img.min() >= 0 and img.max() <= 255 and img.mean() > 1, '{}, min {}, max {}, mean {}'.format(i, img.min(), img.max(), img.mean())
            images.append(img)
            if s == 0: # only get sample_name at the first scale (others are the same)
                sample_names.append('{}_{}_{}.jpg'.format(classIDs[sample_idx], txt_idx, sample_idx))
                sample_idx += 1
                if sample_idx % txt_idx_rest_iter == 0:
                    txt_idx += 1

        print ('read {} with {} images'.format(k, data.shape[0]))
        ms_images[k] = images

    print ('Totally {} images/scale are loaded at scales {}'.format(len(images), ms_images.keys() ))
    assert(len(sample_names) == sample_idx)

    return ms_images, return_path


def preprocess(img):
    # print('img', img.shape, img.max(), img.min())
    # img = Image.fromarray(img, 'RGB')
    if len(img.shape) == 2:
        img = np.resize(img, (img.shape[0], img.shape[1], 3))
    img = scipy.misc.imresize(img, (299, 299, 3),
                              interp='bilinear')
    img = img.astype(np.float32)
    # [0, 255] --> [0, 1] --> [-1, 1]
    img = img / 127.5 - 1.
    # print('img', img.shape, img.max(), img.min())
    return np.expand_dims(img, 0)
