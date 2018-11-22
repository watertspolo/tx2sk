# coding: utf-8
import argparse
import h5py
import join
from datetime import datetime

import os
import sys
sys.path.append('/home/shigaki/code/skip_thoughts/sent2vec/')
from training import loading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file', type=str, default='../../Results/sample_captions.txt',
                        help='caption file')
    parser.add_argument('--data_dir', type=str, default='../../Results',
                        help='Data Directory')

    args = parser.parse_args()

    now = datetime.now()
    result_dir = os.path.join(args.data_dir, '{0:%Y%m%d.%H:%M}'.format(now))
    print result_dir

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        os.makedirs('{}.v2'.format(result_dir))
        result_dir = '{}.v2'.format(result_dir)

    with open(args.caption_file) as f:
        captions = f.read().split('\n')
        # mecab_wakati = MeCab.Tagger('-Owakati')
        # captions = []
        # for text in captions_normal:
        # 	words = mecab_wakati.parse(text)
        # 	sent = ' '.join(words)
        # 	captions.append(sent)

    captions = [cap for cap in captions if len(cap) > 0]

    """モデルのロード"""
    loading.path_to_dictionary = '/home/shigaki/code/skip_thoughts/sent2vec/models/caption/dictionary_cap.pkl'
    loading.path_to_model = '/home/shigaki/code/skip_thoughts/sent2vec/models/caption/model_cap.npz'
    model = loading.load_model()

    caption_vectors = loading.encode(model, captions)

    if os.path.isfile(os.path.join(result_dir, 'sample_caption_vectors.hdf5')):
        os.remove(os.path.join(result_dir, 'sample_caption_vectors.hdf5'))
    h = h5py.File(os.path.join(result_dir, 'sample_caption_vectors.hdf5'))
    h.create_dataset('vectors', data=caption_vectors)
    h.close()

if __name__ == '__main__':
    main()
