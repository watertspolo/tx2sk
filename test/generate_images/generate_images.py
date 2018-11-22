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
from .proj_utils.local_utils import *
from .proj_utils.torch_utils import *
from .HDGan import to_img_dict_
from PIL import Image, ImageDraw, ImageFont
import functools
import time, json, h5py


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='batch size.')
    parser.add_argument('--device_id', type=int, default=0,
                        help='which device')
    parser.add_argument('--load_from_epoch', type=int, default=0,
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--captions',    type=str,      default=None,
                        help='which dataset to use [birds or flowers]')
    parser.add_argument('--noise_dim', type=int, default=100, metavar='N',
                        help='the dimension of noise.')
    parser.add_argument('--finest_size', type=int, default=256, metavar='N',
                        help='target image size.')
    parser.add_argument('--test_sample_num', type=int, default=None,
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--save_visual_results', action='store_true',
                        help='if save visual results in folders')

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    if args.finest_size <= 256:
        netG = Generator(sent_dim=1024, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, num_resblock=1)
    else:
        from HDGan.models.hd_networks import GeneratorSuperL1Loss
        netG = GeneratorSuperL1Loss(sent_dim=1024, noise_dim=args.noise_dim, emb_dim=128, hid_dim=128, num_resblock=2)

    caption_vectors = os.path.join(save_root, args.captions)

    device_id = getattr(args, 'device_id', 0)

    if args.cuda:
        netG = netG.cuda(device_id)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    model_name = args.model_name

    save_folder = os.path.join(save_root, args.dataset, model_name + '_testing_num_{}'.format(args.test_sample_num))

    if save_folder:
        print('すでにフォルダが存在します' + str(save_folder))
    else:
        mkdirs(save_folder)

    generate_images(caption_vectors, model_root, model_name, save_folder, netG, args)


def generate_images(dataset, model_root, model_name, save_folder, netG, args):

	h = h5py.File( args.caption_thought_vectors )
	caption_vectors = np.array(h['vectors'])
	caption_image_dic = {}
	for cn, caption_vector in enumerate(caption_vectors):

		caption_images = []
		z_noise = np.random.uniform(-1, 1, [args.n_images, args.z_dim])
		caption = [ caption_vector[0:args.caption_vector_length] ] * args.n_images

		[ gen_image ] = sess.run( [ outputs['generator'] ],
			feed_dict = {
				input_tensors['t_real_caption'] : caption,
				input_tensors['t_z'] : z_noise,
			} )

		caption_images = [gen_image[i,:,:,:] for i in range(0, args.n_images)]
		caption_image_dic[ cn ] = caption_images
		print "Generated", cn

	for f in os.listdir( join(args.data_dir, 'val_samples')):
		if os.path.isfile(f):
			os.unlink(join(args.data_dir, 'val_samples/' + f))

	for cn in range(0, len(caption_vectors)):
		caption_images = []
		for i, im in enumerate( caption_image_dic[ cn ] ):

			caption_images.append( im )
			caption_images.append( np.zeros((256, 5, 3)) )
		combined_image = np.concatenate( caption_images[0:-1], axis = 1 )
		scipy.misc.imsave( join(args.data_dir, 'val_samples/combined_image_{}.jpg'.format(cn+1)) , combined_image)


if __name__ == '__main__':
	main()
