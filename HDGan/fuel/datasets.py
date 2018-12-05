# -*- coding: utf-8 -*-
from .datasets_basic import Dataset as BasicDataset
from .datasets_multithread import COCODataset
from .datasets_multithread import Dataset as BasicCOCODataset


def Dataset(data_name, datadir, img_dir, img_size, batch_size, cap_name, emb_file, n_embed, mode, multithread=True):

    # we don't create multithread loader for bird and flower
    # because we need to make sure the `wrong' images should be in different classes
    # with the real images. It is hard to guarantee that in the parallel mode.
    if 'birds' in data_name or 'flower' in data_name:
        return BasicDataset(datadir, img_size, batch_size, n_embed, mode)
    elif 'coco' in data_name:
        if mode == 'test':
            mode = 'val'
        if not multithread:
            # we do not need parallel in testing
            print('mode = {} Basic'.format(mode))
            return BasicCOCODataset(datadir, img_dir=img_dir, img_size=img_size, batch_size=batch_size, cap_name=cap_name, emb_file=emb_file, n_embed=n_embed, mode=mode)
        else:
            print('mode = {}'.format(mode))
            return COCODataset(datadir, img_dir, img_size, batch_size, cap_name, emb_file, n_embed, mode, threads=4).load_data()
