# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------#
# dataloader for birds and flowers is modified from https://github.com/hanzhanggit/StackGAN
# don't set batch size 1
#-------------------------------------------------------------------------#
import numpy as np
import h5py
import glob
import random
from collections import OrderedDict
import sys, os
import scipy.misc as misc
import torch.utils.data
from functools import partial
from .datasets_basic import resize_images


def img_loader_func(img_names, imgpath=None, img_size=256):  # imgpath=None
    res = []
    error = 0
    for i_n in img_names:
        try:
            img = misc.imread(os.path.join(imgpath, i_n))
            img = misc.imresize(img, (img_size, img_size))
            if len(img.shape) != 3:
                # グレー画像の場合
                img = np.tile(img[:, :, np.newaxis], [1, 1, 3])

            res.append(img[np.newaxis, :, :, :])
        except FileNotFoundError:
            error += 1
            continue
    res = np.concatenate(res, axis=0)

    return res


class Dataset(object):
    def __init__(self, workdir, img_dir, img_size, batch_size, cap_name, emb_file, n_embed, mode='train'):

        if img_size in [256, 512]:
            self.output_res = [64, 128, 256]
            if img_size == 512: self.output_res += [512]
        elif img_size in [64]:
            self.output_res = [64]

        self.embedding_filename = emb_file
        self.image_shape = [img_size, img_size, 3]

        self.batch_size = batch_size
        self.cap_name = cap_name
        self.n_embed = n_embed

        self.imsize = img_size
        self.workdir = workdir
        self.img_dir = img_dir
        self.train_mode = mode == 'train'
        self.get_data(self.workdir)

        self._text_index = 0
        self.text_index = 0
        self._saveIDs = np.arange(self._num_examples)
        self._classIDs = np.zeros(self._num_examples)
        print('>> Init COCO data loader ', mode)
        print('\t {} samples (batch_size = {})'.format(self._num_examples, self.batch_size))
        print('\t {} output resolutions'.format(self.output_res))
        print ('\t {} embeddings used'.format(n_embed))

    def get_data(self, data_dir):

        # data_root = os.path.split(data_dir)[0]

        """訓練画像のパスを取得"""
        if self.train_mode:
            img_path = os.path.join(data_dir, self.img_dir)
            # img_path = os.path.join('/home/shigaki/Data/images/MSCOCO_2017train', 'train2017')
        # else:
        #     img_path = os.path.join('/home/shigaki/Data/images/MSCOCO_2017train', 'val2017')

        self.images = partial(img_loader_func, imgpath=img_path, img_size=self.imsize)

        """filename, caption, embedding のパスを取得"""
        self.embeddings_h = h5py.File(os.path.join(data_dir, self.embedding_filename))

        '''filename'''
        self.filenames = []
        for filename in self.embeddings_h:
            self.filenames.append(filename)
        print('読み込んだキャプションファイルの数(画像単位): '+str(len(self.filenames)))

        '''キャプション'''
        # self.captions = OrderedDict()
        # error = 0
        #
        # # キャプションファイルのパスとファイル名をリストで取得
        # cap_path = os.path.join(data_dir, 'STAIR-Captions/inter/{}/*.txt'.format(self.cap_name))
        # caption_files = glob.glob(cap_path)
        # filename_list = [os.path.basename(r) for r in caption_files]  # ファイル名だけを抽出
        #
        # # key:画像ファイル名、　value:キャプション　で辞書作成
        # for cap_file in filename_list:
        #     try:
        #         with open(os.path.join(data_dir, 'STAIR-Captions/inter/{}/{}'.format(self.cap_name, cap_file)), 'r') as f:
        #             captions = f.readlines()
        #
        #             if len(os.path.splitext(cap_file)[0]) == 7:
        #                 img_file = "00000" + os.path.splitext(cap_file)[0] + ".jpg"
        #             elif len(os.path.splitext(cap_file)[0]) == 6:
        #                 img_file = "000000" + os.path.splitext(cap_file)[0] + ".jpg"
        #             elif len(os.path.splitext(cap_file)[0]) == 5:
        #                 img_file = "0000000" + os.path.splitext(cap_file)[0] + ".jpg"
        #             elif len(os.path.splitext(cap_file)[0]) == 4:
        #                 img_file = "00000000" + os.path.splitext(cap_file)[0] + ".jpg"
        #             elif len(os.path.splitext(cap_file)[0]) == 3:
        #                 img_file = "000000000" + os.path.splitext(cap_file)[0] + ".jpg"
        #             elif len(os.path.splitext(cap_file)[0]) == 2:
        #                 img_file = "0000000000" + os.path.splitext(cap_file)[0] + ".jpg"
        #             elif len(os.path.splitext(cap_file)[0]) == 1:
        #                 img_file = "00000000000" + os.path.splitext(cap_file)[0] + ".jpg"
        #
        #             # １画像につき5キャプ地温
        #             self.captions[img_file] = captions
        #
        #     except:
        #         error += 1
        #         pass
        # print('キャプションファイル数: ', len(self.captions))
        # print('存在しないセット数: ', error)

        '''embedding'''
        '''
        self.embeddings = []
        for i, emb_filename in enumerate(self.filenames):
            self.embeddings.append(np.array(h[emb_filename]))
        self.embedding_shape = len(self.embeddings[0][0])
        print('読み込んだembeddingの次元数: ', self.embedding_shape)
        '''
        """ファイル数"""
        self._num_examples = len(self.filenames)

    # def readCaptions(self, filename):
    #     cap = self.captions[filename]
    #     return cap

    def transform(self, images):
        '''imsize=256'''
        transformed_images = np.zeros([images.shape[0], self.imsize, self.imsize, 3])
        ori_size = images.shape[1]
        for i in range(images.shape[0]):
            if self.train_mode:
                h1 = int(np.floor((ori_size - self.imsize) * np.random.random()))
                w1 = int(np.floor((ori_size - self.imsize) * np.random.random()))
            else:
                # center crop
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
            # sampled_captions = []
            for i in range(batch_size):
                randix = np.random.choice(embedding_num,
                                          sample_num, replace=False)
                if sample_num == 1:
                    randix = int(randix)
                    # captions = self.readCaptions(filenames[i])
                    # sampled_captions.append(captions[randix])
                    sampled_embeddings.append(embeddings[i, randix, :])
                else:
                    e_sample = embeddings[i, randix, :]
                    e_mean = np.mean(e_sample, axis=0)
                    # captions = self.readCaptions(filenames[i])
                    # sampled_captions.append(captions[randix[0]])

                    sampled_embeddings.append(e_mean)
            sampled_embeddings_array = np.array(sampled_embeddings)

            # return np.squeeze(sampled_embeddings_array)
            return np.squeeze(sampled_embeddings_array)#, sampled_captions

    def __getitem__(self, index):
        """Return the next `batch_size` examples from this data set."""
        batch_size = self.batch_size

        start = self.text_index
        if (start + batch_size) > self._num_examples:
            end = self._num_examples
            self.text_index = 0
        else:
            end = start + batch_size
        self.text_index += batch_size

        current_ids = range(start, end)
        # print('__getitem__(index):' + str(index))
        # current_ids = [index]  # only take one
        fake_ids = np.random.randint(self._num_examples, size=len(current_ids))

        images_dict = OrderedDict()
        wrongs_dict = OrderedDict()

        """順番どおりとランダムの２種類のリストを用意"""
        filenames = [self.filenames[i] for i in current_ids]
        fake_filenames = [self.filenames[i] for i in fake_ids]

        """対応する画像を読み込みリサイズ"""
        sampled_images = self.images(filenames)
        sampled_wrong_images = self.images(fake_filenames)
        sampled_images = np.array(sampled_images, dtype=np.float32)
        # sampled_images = sampled_images.astype(np.float32)
        sampled_wrong_images = np.array(sampled_wrong_images, dtype=np.float32)
        # sampled_wrong_images = sampled_wrong_images.astype(np.float32)

        sampled_images = self.transform(sampled_images)
        sampled_wrong_images = self.transform(sampled_wrong_images)

        images_dict = {}
        wrongs_dict = {}

        """64,128,256のサイズごとに辞書に格納"""
        for size in self.output_res:
            tmp = resize_images(sampled_images, shape=[size, size]).transpose((0, 3, 1, 2))
            # tmp = (len(self.filenames), 3, size, size)
            '''0-1の範囲に正規化'''
            tmp = tmp * (2. / 255) - 1.
            # try:
            #     tmp = np.squeeze(tmp, 0)
            # except:
            #     continue
            # print('error='+str(error))
            images_dict['output_{}'.format(size)] = np.array(tmp, dtype=np.float32)
            # images_dict['output_{}'.format(size)] = tmp.astype(np.float32)
            tmp = resize_images(sampled_wrong_images, shape=[size, size]).transpose((0, 3, 1, 2))
            tmp = tmp * (2. / 255) - 1.
            # try:
            #     tmp = np.squeeze(tmp, 0)
            # except:
            #     continue
            # print('error2='+str(error2))
            wrongs_dict['output_{}'.format(size)] = np.array(tmp, dtype=np.float32)
            # wrongs_dict['output_{}'.format(size)] = tmp.astype(np.float32)

        ret_list = [images_dict, wrongs_dict]
        embeddings = []
        for i, emb_filename in enumerate(filenames):
            feature = np.array(self.embeddings_h[emb_filename])
            embeddings.append(feature)

        embeddings = np.array(embeddings)
        sampled_embeddings = self.sample_embeddings(embeddings, filenames, self.n_embed)
        # sampled_embeddings, sampled_captions = self.sample_embeddings(embeddings, filenames, self.n_embed)

        # sampled_embeddings, sampled_captions = self.sample_embeddings(embeddings[current_ids],
        #                                                               filenames, self.n_embed)
        ret_list.append(sampled_embeddings)
        # ret_list.append(sampled_captions)
        #
        # ret_list.append(filenames)

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

        current_ids = range(start, end)
        sampled_filenames = [self.filenames[i] for i in current_ids]
        print('next_batch_test:'+str(sampled_filenames))

        sampled_images = np.array(self.images(sampled_filenames), dtype=np.float32)
        # sampled_images = self.images(sampled_filenames).astype(np.float32)
        sampled_images = self.transform(sampled_images)
        sampled_images = sampled_images * (2. / 255) - 1.

        test_filenames = self.filenames[start:end]
        sampled_embeddings = []
        for i, emb_filename in enumerate(test_filenames):
            feature = np.array(self.embeddings_h[emb_filename])
            sampled_embeddings.append(feature)
        _, embedding_num, _ = sampled_embeddings.shape
        sampled_embeddings_batchs = []

        # sampled_captions = []
        # for i in range(len(sampled_filenames)):
        #     captions = self.readCaptions(sampled_filenames[i])
        #     sampled_captions.append(captions)

        for i in range(np.minimum(max_captions, embedding_num)):
            batch = sampled_embeddings[:, i, :]
            sampled_embeddings_batchs.append(batch)

        return [sampled_images, sampled_embeddings_batchs, self._saveIDs[start:end], self._classIDs[start:end]]
        # return [sampled_images, sampled_embeddings_batchs, sampled_captions, self._saveIDs[start:end], self._classIDs[start:end]]

    def __len__(self):
        return self._num_examples


class COCODataset():
    def __init__(self, data_dir, img_dir, img_size, batch_size, cap_name, emb_file, num_embed, mode='train', threads=1, drop_last=True):
        print ('>> create multithread loader with {} threads ...'.format(threads))
        self.dataset = Dataset(data_dir, img_dir=img_dir, img_size=img_size, batch_size=batch_size, cap_name=cap_name, emb_file=emb_file, n_embed=num_embed, mode=mode)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=mode == 'train',
            num_workers=threads,
            drop_last=drop_last)

        self.dataloader._num_examples = len(self.dataset)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
