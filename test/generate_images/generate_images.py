# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.join('..', '..'))
proj_root = os.path.join('..', '..')
data_root = os.path.join('../../../../..', 'Data')
model_root = os.path.join(proj_root, 'Models')
save_root = os.path.join(proj_root, 'Results')

import numpy as np
import argparse, os
import torch, h5py
import torch.nn as nn
from collections import OrderedDict

from HDGan.proj_utils.local_utils import mkdirs
from HDGan.HDGan_test import test_gans
from HDGan.fuel.datasets import Dataset

from HDGan.models.hd_networks import Generator

import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter

from torch.nn.utils import clip_grad_norm
from .proj_utils.plot_utils import *
# from .proj_utils.local_utils import *
from .proj_utils.torch_utils import *
from .HDGan import to_img_dict_
from .HDGan import save_imgs
from PIL import Image, ImageDraw, ImageFont
import functools
import time, json, h5py


def generate_images(caption_vectors, model_root, model_name, save_folder, netG, args):
    # helper function
    netG.eval()

    model_folder = os.path.join(model_root, model_name)

    ''' load model '''
    assert args.load_from_epoch != '', 'args.load_from_epoch is empty'
    G_weightspath = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))
    print('reload weights from {}'.format(G_weightspath))
    weights_dict = torch.load(G_weightspath, map_location=lambda storage, loc: storage)
    # load_partial_state_dict(netG, weights_dict)
    sample_weight_name = [a for a in weights_dict.keys()][0]
    if 'module' in sample_weight_name:  # if the saved model is wrapped by DataParallel.
        netG = nn.parallel.DataParallel(netG, device_ids=[0])
    # TODO note that strict is set to false for now. It is a bit risky
    netG.load_state_dict(weights_dict, strict=False)

    h = h5py.File(caption_vectors)
    caption_vectors = np.array(h['vectors'])

    to_img_dict = functools.partial(to_img_dict_, super512=args.finest_size == 512)

    for cn, caption_vector in enumerate(caption_vectors):

        z = torch.FloatTensor(args.test_sample_num, args.noise_dim).normal_(0, 1)
        # z = to_device(z)
        z.data.normal_(0, 1)

        cap_vec = [caption_vector[0:args.caption_vector_length]] * args.test_sample_num
        embeddings = cap_vec
        # embeddings = to_device(cap_vec, requires_grad=False)

        fake_images, _ = to_img_dict(netG(embeddings, z))

        for k, sample in fake_images.items():
            save_imgs(sample,
                      args.load_from_epoch, k, 'test_images_{}'.format(cn), path=save_folder, model_name=model_name)
        print("Generated"+str(cn))


def main():

    parser = argparse.ArgumentParser(description='Gans')

    parser.add_argument('--embedding_filename', type=str, default='sample_caption_vectors.hdf5',
                        help='embedding filename.')
    parser.add_argument('--caption_vector_length', type=int, default=2400, metavar='N',
                        help='caption vector length.')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='batch size.')

    parser.add_argument('--device_id', type=int, default=0,
                        help='which device')
    parser.add_argument('--load_from_epoch', type=int, default=70,
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default='image_human256v2')
    parser.add_argument('--dataset',    type=str,      default='coco',
                        help='which dataset to use [birds or flowers]')
    parser.add_argument('--noise_dim', type=int, default=100, metavar='N',
                        help='the dimension of noise.')
    parser.add_argument('--finest_size', type=int, default=256, metavar='N',
                        help='target image size.')
    parser.add_argument('--test_sample_num', type=int, default=4,
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--save_visual_results', action='store_true',
                        help='if save visual results in folders')

    args = parser.parse_args()

    # args.cuda = torch.cuda.is_available()

    if args.finest_size <= 256:
        netG = Generator(sent_dim=1024, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, num_resblock=1)
    else:
        from HDGan.models.hd_networks import GeneratorSuperL1Loss
        netG = GeneratorSuperL1Loss(sent_dim=1024, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, num_resblock=2)

    # device_id = getattr(args, 'device_id', 0)

    # if args.cuda:
    #     netG = netG.cuda(device_id)
    #     import torch.backends.cudnn as cudnn
    #     cudnn.benchmark = True

    caption_vectors = args.embedding_filename
    model_name = args.model_name

    save_folder = os.path.join(save_root, args.dataset, model_name + '_testing_num_{}'.format(args.test_sample_num))

    if save_folder:
        print('すでにフォルダが存在します' + str(save_folder))
    else:
        mkdirs(save_folder)

    generate_images(caption_vectors, model_root, model_name, save_folder, netG, args)


if __name__ == '__main__':
	main()
