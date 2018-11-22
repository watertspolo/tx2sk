# -*- coding: utf-8 -*-
import argparse
import torch
# from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
import os
import sys
sys.path.insert(0, os.path.join('..', '..'))

proj_root = os.path.join('..', '..')
data_root = os.path.join(proj_root, 'Data')
model_root = os.path.join(proj_root, 'Models')

import torch.nn as nn
from collections import OrderedDict

from HDGan.models.hd_networks import Generator
from HDGan.models.hd_networks import Discriminator

from HDGan.HDGan import train_gans
from HDGan.fuel.datasets import Dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gans')

    parser.add_argument('--data_dir', type=str,
                        default='/home/shigaki/Data', help='data directory')
    parser.add_argument('--img_dir', type=str,
                        default='images/MSCOCO_2017train_sketch', help='imgdata directory')
    parser.add_argument('--cap_name', type=str,
                        default='cocohuman_2017', help='cap dirctory--Data/STAIR-Caption/inter/()')
    parser.add_argument('--embedding_filename', type=str,
                        default='cap_vec_human.hdf5', help='embedding filename')

    parser.add_argument('--reuse_weights', action='store_true',
                        default=False, help='continue from last checkout point')
    parser.add_argument('--load_from_epoch', type=int,
                        default=0,  help='load from epoch')

    parser.add_argument('--batch_size', type=int,
                        default=8, metavar='N', help='batch size.')
    parser.add_argument('--sent_dim', type=int,
                        default=2400, metavar='N', help='dimention of skip-thought vectors')
    parser.add_argument('--device_id',  type=int,
                        default=0,  help='which device')

    parser.add_argument('--model_name', type=str,      default='sketch_human256')
    parser.add_argument('--dataset',    type=str,      default='coco',
                        help='which dataset to use [birds or flowers]')

    parser.add_argument('--num_resblock', type=int, default=1,
                        help='number of resblock in generator')
    parser.add_argument('--epoch_decay', type=float, default=100,
                        help='decay learning rate by half every epoch_decay')
    parser.add_argument('--finest_size', type=int, default=256,
                        metavar='N', help='target image size.')
    parser.add_argument('--init_256generator_from', type=str,  default='')
    parser.add_argument('--maxepoch', type=int, default=600, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--g_lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--d_lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--save_freq', type=int, default=5, metavar='N',
                        help='how frequent to save the model')
    parser.add_argument('--display_freq', type=int, default=200, metavar='N',
                        help='plot the results every {} batches')
    parser.add_argument('--verbose_per_iter', type=int, default=200,
                        help='save images per iteration')
    parser.add_argument('--log_inter', type=int, default=40,
                        help='print losses per iteration')

    parser.add_argument('--num_emb', type=int, default=5, metavar='N',
                        help='number of emb chosen for each image during training.')
    parser.add_argument('--noise_dim', type=int, default=100, metavar='N',
                        help='the dimension of noise.')
    parser.add_argument('--ncritic', type=int, default=1, metavar='N',
                        help='the channel of each image.')
    parser.add_argument('--test_sample_num', type=int, default=4,
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--KL_COE', type=float, default=4, metavar='N',
                        help='kl divergency coefficient.')
    # parser.add_argument('--visdom_port', type=int, default=8097,
    #                     help='The port should be the same with the port when launching visdom')
    parser.add_argument('--gpus', type=str, default='0',
                        help='which gpu')
    # add more
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print(args)

    '''Generator'''
    if args.finest_size <= 256:
        netG = Generator(sent_dim=args.sent_dim, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, num_resblock=1)
    else:
        # 512サイズの画像を生成する場合のモデル定義
        assert args.init_256generator_from != '', '256 generator need to be intialized'
        from HDGan.models.hd_networks import GeneratorSuperL1Loss
        netG = GeneratorSuperL1Loss(sent_dim=args.sent_dim, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, num_resblock=2, G256_weightspath=args.init_256generator_from)
    '''Discriminator'''
    netD = Discriminator(num_chan=3, hid_dim=128, sent_dim=args.sent_dim, emb_dim=128)

    '''GPU使用'''
    gpus = [a for a in range(len(args.gpus.split(',')))]
    torch.cuda.set_device(gpus[0])
    args.batch_size = args.batch_size * len(gpus)
    if args.cuda:
        print ('>> Parallel models in {} GPUS'.format(gpus))
        netD = nn.parallel.DataParallel(netD, device_ids=range(len(gpus)))
        netG = nn.parallel.DataParallel(netG, device_ids=range(len(gpus)))

        netD = netD.cuda()
        netG = netG.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    """訓練データ定義"""
    data_name = args.dataset
    datadir = args.data_dir

    dataset_train = Dataset(data_name, datadir, img_dir=args.img_dir, img_size=args.finest_size, batch_size=args.batch_size,
                            cap_name=args.cap_name, emb_file=args.embedding_filename, n_embed=args.num_emb, mode='train')
    # dataset_test = Dataset(datadir, img_size=args.finest_size,
    #                        batch_size=args.batch_size, n_embed=1, mode='test')
    # 今回はtrainだけでlossを表示するのでコメントアウトした

    # for images, wrong_images, np_embeddings, _, _ in dataset_train:
    #     for key in [64, 128, 256]:
    #         print(images[key])
    #         print(wrong_images[key])
    #         print(np_embeddings[key])
    #     break

    """訓練開始"""
    print('>> 訓練開始 ...')
    model_name = '{}_{}'.format(args.model_name, data_name)

    train_gans(dataset_train, model_root, model_name, netG, netD, args)  # (dataset_train, dataset_test) or (dataset_train)
