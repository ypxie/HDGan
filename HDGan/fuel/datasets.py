from .datasets_basic import Dataset as BasicDataset
from .datasets_multithread import COCODataset
from .datasets_multithread import Dataset as BasicCOCODataset


def Dataset(datadir, img_size, batch_size, n_embed, mode):

    # we don't create multithread loader for bird and coco 
    # because we need to make sure the wrong images should be in different classes 
    # with the real images. It is hard to guarantee in parallel mode.
    if 'birds' in datadir or 'flower' in datadir:
        return BasicDataset(datadir, img_size, batch_size, n_embed, mode)
    elif 'coco' in datadir:
        if mode == 'test' or mode == 'val':
            return BasicCOCODataset(datadir, img_size=img_size, batch_size=batch_size, n_embed=n_embed, mode='val')
        else:
            return COCODataset(datadir, img_size, batch_size, n_embed, mode, threads=0).load_data()