from .datasets_basic import Dataset as BasicDataset
from .datasets_multithread import COCODataset
from .datasets_multithread import Dataset as BasicCOCODataset


def Dataset(datadir, img_size, batch_size, n_embed, mode, multithread=True):

    # we don't create multithread loader for bird and flower 
    # because we need to make sure the `wrong' images should be in different classes 
    # with the real images. It is hard to guarantee that in the parallel mode.
    if 'birds' in datadir or 'flower' in datadir:
        return BasicDataset(datadir, img_size, batch_size, n_embed, mode)
    elif 'coco' in datadir:
        if mode == 'test': mode = 'val'
        if not multithread:
            # we do not need parallel in testing
            return BasicCOCODataset(datadir, img_size=img_size, batch_size=batch_size, n_embed=n_embed, mode=mode)
        else:
            return COCODataset(datadir, img_size, batch_size, n_embed, mode, threads=2).load_data()