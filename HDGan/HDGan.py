# coding: utf-8
import numpy as np
import pandas as pd
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter
from torchvision.utils import save_image

from torch.nn.utils import clip_grad_norm
# from .proj_utils.plot_utils import *
from .proj_utils.torch_utils import *
import tqdm
import time
import json
import functools


def to_img_dict_(*inputs, super512 = False):

    if type(inputs[0]) == tuple:
        inputs = inputs[0]
    res = {}
    res['output_64'] = inputs[0]
    res['output_128'] = inputs[1]
    res['output_256'] = inputs[2]
    # generator returns different things for 512HDGAN
    if not super512:
        # from Generator
        mean_var = (inputs[3], inputs[4])
        loss = mean_var
    else:
        # from GeneratorL1Loss of 512HDGAN
        res['output_512'] = inputs[3]
        l1loss = inputs[4]  # l1 loss
        loss = l1loss

    return res, loss


def get_KL_Loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    kld = mu.pow(2).add(logvar.mul(2).exp()).add(-1).mul(0.5).add(logvar.mul(-1))
    kl_loss = torch.mean(kld)
    return kl_loss


def compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels):

    criterion = nn.MSELoss()
    real_d_loss = criterion(real_logit, real_labels)
    wrong_d_loss = criterion(wrong_logit, fake_labels)
    fake_d_loss = criterion(fake_logit, fake_labels)

    discriminator_loss = real_d_loss + (wrong_d_loss+fake_d_loss) / 2.
    return discriminator_loss


def compute_d_img_loss(wrong_img_logit, real_img_logit, fake_img_logit, real_labels, fake_labels):

    criterion = nn.MSELoss()
    wrong_d_loss = criterion(wrong_img_logit, real_labels)
    real_d_loss = criterion(real_img_logit, real_labels)
    fake_d_loss = criterion(fake_img_logit, fake_labels)

    return fake_d_loss + (wrong_d_loss+real_d_loss) / 2


def compute_g_loss(fake_logit, real_labels):

    criterion = nn.MSELoss()
    generator_loss = criterion(fake_logit, real_labels)
    return generator_loss


# def plot_imgs(samples, epoch, typ, name, path, model_name=None):
#
#     tmpX = save_images(samples, save=not path == '', save_path=os.path.join(
#         path, '{}_epoch{}_{}.png'.format(name, epoch, typ)), dim_ordering='th')
#     plot_img(X=tmpX, win='{}_{}.png'.format(name, typ), env=model_name)


def save_timgs(samples, typ, name, path, model_name=None):
    samples = torch.Tensor(samples)
    # print(sample.shape)
    save_image(samples,
               os.path.join(path,
                            '{}_{}.png'.format(name, typ)),
               normalize=True, range=(-1, 1))
    # fake_samples = fake_samples.data.cpu()


def save_imgs(fake_samples, epoch, typ, name, path, model_name=None):
    save_image(fake_samples.data,
               os.path.join(path,
                            '{}_fake_epoch{}_{}.png'.format(name, epoch, typ)),
               normalize=True, range=(-1, 1))


def train_gans(dataset, model_root, model_name, netG, netD, args):
    """
    Parameters:
    ----------
    dataset:
        data loader. refers to fuel.dataset
    model_root:
        the folder to save the model weights
    model_name :
        the model_name
    netG:
        Generator
    netD:
        Descriminator
    """
    model_folder = os.path.join(model_root, model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    log_path = os.path.join(model_folder, 'log')
    logger = getLogger(__name__)
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(log_path, 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    logger.debug('start')

    d_lr = args.d_lr
    g_lr = args.g_lr
    tot_epoch = args.maxepoch

    """sampler定義"""
    train_sampler = iter(dataset)
    # test_sampler = iter(dataset)

    updates_per_epoch = int(dataset._num_examples / args.batch_size)

    """optimizerの設定"""
    parameters_D = filter(lambda x: x.requires_grad, netD.parameters())
    optimizerD = optim.Adam(parameters_D, lr=d_lr, betas=(0.5, 0.999))
    # optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=g_lr, betas=(0.5, 0.999))

    """途中から訓練を再開する場合"""
    if args.reuse_weights:
        D_weightspath = os.path.join(
            os.path.join(model_root, model_name), 'D_epoch{}.pth'.format(args.load_from_epoch))
        G_weightspath = os.path.join(
            os.path.join(model_root, model_name), 'G_epoch{}.pth'.format(args.load_from_epoch))

        if os.path.exists(D_weightspath) and os.path.exists(G_weightspath):
            weights_dict = torch.load(
                D_weightspath, map_location=lambda storage, loc: storage)
            print('reload weights from {}'.format(D_weightspath))
            netD_ = netD.module if 'DataParallel' in str(type(netD)) else netD
            netD_.load_state_dict(weights_dict, strict=False)

            print('reload weights from {}'.format(G_weightspath))
            weights_dict = torch.load(
                G_weightspath, map_location=lambda storage, loc: storage)
            netG_ = netG.module if 'DataParallel' in str(type(netG)) else netG
            netG_.load_state_dict(weights_dict, strict=False)

            start_epoch = args.load_from_epoch + 1
            d_lr /= 2 ** (start_epoch // args.epoch_decay)
            g_lr /= 2 ** (start_epoch // args.epoch_decay)
        else:
            raise ValueError('{} or {} do not exist'.format(D_weightspath, G_weightspath))
    else:
        start_epoch = 1

    """lossの可視化準備(dic定義)"""
    # if finest_size == 128:
    #     Din_size[0], Din_size[1] = finest_size/2, finest_size
    # elif finest_size == 256:
    #     Din_size[0], Din_size[1], Din_size[2] = finest_size/4, finest_size/2, finest_size
    # elif finest_size == 512:
    #     Din_size[0], Din_size[1], Din_size[2], Din_size[3] = finest_size/8, finest_size/4, finest_size/2, finest_size

    '''           D or G        each_size or all      pair or image      losses per epoch           '''
    '''                                           or discriminator_loss                             '''
    log_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    #--------Generator niose placeholder used for testing------------#
    z = torch.FloatTensor(args.batch_size, args.noise_dim).normal_(0, 1)
    z = to_device(z)
    # generate a set of fixed test samples to visualize changes in training epoches
    #
    # fixed_images, _, fixed_embeddings, _, _ = next(test_sampler)
    # fixed_embeddings = to_device(fixed_embeddings)
    # fixed_z_data = [torch.FloatTensor(args.batch_size, args.noise_dim).normal_(
    #     0, 1) for _ in range(args.test_sample_num)]
    # fixed_z_list = [to_device(a) for a in fixed_z_data]


    # create discrimnator label placeholder (not a good way)
    REAL_global_LABELS = Variable(torch.FloatTensor(args.batch_size, 1).fill_(1)).cuda()
    FAKE_global_LABELS = Variable(torch.FloatTensor(args.batch_size, 1).fill_(0)).cuda()
    REAL_local_LABELS = Variable(torch.FloatTensor(args.batch_size, 1, 5, 5).fill_(1)).cuda()
    FAKE_local_LABELS = Variable(torch.FloatTensor(args.batch_size, 1, 5, 5).fill_(0)).cuda()

    def get_labels(logit):
        # get discriminator labels for real and fake
        if logit.size(-1) == 1:
            return REAL_global_LABELS.view_as(logit), FAKE_global_LABELS.view_as(logit)
        else:
            return REAL_local_LABELS.view_as(logit), FAKE_local_LABELS.view_as(logit)

    to_img_dict = functools.partial(to_img_dict_, super512=args.finest_size == 512)


    """訓練"""
    for epoch in range(start_epoch, tot_epoch):
        logcount = 0
        start_timer = time.time()
        print('epoch'+str(epoch)+' start...')

        '''訓練が進むに連れて学習率を低下させる'''
        if epoch % args.epoch_decay == 0:
            d_lr = d_lr/2
            g_lr = g_lr/2

            set_lr(optimizerD, d_lr)
            set_lr(optimizerG, g_lr)

        '''StopIterationを避けるためリセット'''
        train_sampler = iter(dataset)
        # test_sampler = iter(dataset[1])

        netG.train()
        netD.train()

        '''batch_size毎'''
        for it in range(updates_per_epoch):
            ncritic = args.ncritic

            # '''画像のチャネル毎(1)'''
            # for _ in range(ncritic):
            """"""

            '''サンプルデータ'''
            try:
                images, wrong_images, np_embeddings = next(train_sampler)
                # images, wrong_images, np_embeddings, _, _ = next(train_sampler)
            except:
                train_sampler = iter(dataset)  # reset
                images, wrong_images, np_embeddings = next(train_sampler)
                # images, wrong_images, np_embeddings, _, _ = next(train_sampler)

            embeddings = to_device(np_embeddings, requires_grad=False)
            z.data.normal_(0, 1)

            ''' update D '''
            for p in netD.parameters(): p.requires_grad = True
            netD.zero_grad()  # 勾配の初期化

            if embeddings.shape[0] != args.batch_size:
                break

            fake_images, mean_var = to_img_dict(netG(embeddings, z))

            discriminator_loss = 0

            '''サイズの異なる画像を繰り返し処理する(64,128,256)'''
            for key, _ in fake_images.items():
                this_img = to_device(images[key])
                this_wrong = to_device(wrong_images[key])
                this_fake = Variable(fake_images[key].data)

                real_logit,  real_img_logit_local = netD(this_img, embeddings)
                wrong_logit, wrong_img_logit_local = netD(this_wrong, embeddings)
                fake_logit,  fake_img_logit_local = netD(this_fake, embeddings)

                '''Dのpair_lossを計算'''
                real_labels, fake_labels = get_labels(real_logit)
                pair_loss = compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels)

                '''Dのimage_lossを計算'''
                real_labels, fake_labels = get_labels(real_img_logit_local)
                img_loss = compute_d_img_loss(wrong_img_logit_local, real_img_logit_local, fake_img_logit_local, real_labels, fake_labels)

                '''pair_lossとimage_lossの合計をD_lossとして定義'''
                discriminator_loss += (pair_loss + img_loss)

                '''epoch単位でlossをlog_dictに格納'''
                if it == args.log_inter:
                    pair_loss_num = to_numpy(pair_loss).mean()
                    log_dict['D'][key]['pair_loss'].append(pair_loss_num)

                    img_loss_num = to_numpy(img_loss).mean()
                    log_dict['D'][key]['image_loss'].append(img_loss_num)

            discriminator_loss.backward()  # 逆伝搬
            optimizerD.step()  # 重み更新
            netD.zero_grad()  # 勾配初期化

            '''それぞれの画像サイズにおけるD_lossの平均を定義'''
            if it == args.log_inter:
                a=[]
                d_loss_val = to_numpy(discriminator_loss).mean()
                log_dict['D']['all']['d_loss_val'].append(d_loss_val)

            ''' update G '''
            for p in netD.parameters():p.requires_grad = False  # to avoid computation
            netG.zero_grad()  # 勾配の初期化

            # TODO Test if we do need to sample again in Birds and Flowers
            # z.data.normal_(0, 1)  # resample random noises
            # fake_images, kl_loss = netG(embeddings, z)

            loss_val = 0
            if type(mean_var) == tuple:
                '''平均と分散からkl_lossを取得・平均後、発散係数KL_COE(4)を乗算してG_lossを計算'''
                kl_loss = get_KL_Loss(mean_var[0], mean_var[1])
                kl_loss_val = to_numpy(kl_loss).mean()
                generator_loss = args.KL_COE * kl_loss
            else:
                # when trian 512HDGAN. KL loss is fixed since we assume 256HDGAN is trained.
                # mean_var actually returns pixel-wise l1 loss (see paper)
                generator_loss = mean_var

            '''サイズの異なる画像を繰り返し処理する(64,128,256)'''
            for key, _ in fake_images.items():
                this_fake = fake_images[key]
                fake_pair_logit, fake_img_logit_local = netD(this_fake, embeddings)

                '''Gのpair_lossを計算'''
                real_labels, _ = get_labels(fake_pair_logit)
                generator_loss += compute_g_loss(fake_pair_logit, real_labels)

                '''Gのimage_lossを計算'''
                real_labels, _ = get_labels(fake_img_logit_local)
                img_loss = compute_g_loss(fake_img_logit_local, real_labels)

                '''epoch単位でlossをlog_dictに格納'''
                if it == args.log_inter:
                    pair_loss_num = to_numpy(generator_loss).mean()
                    log_dict['G'][key]['pair_loss'].append(pair_loss_num)

                '''pair_lossとimage_lossの合計をG_lossとして定義'''
                generator_loss += img_loss

                '''epoch単位でlossをlog_dictに格納'''
                if it == args.log_inter:
                    img_loss_num = to_numpy(img_loss).mean()
                    log_dict['G'][key]['image_loss'].append(img_loss_num)

            '''pair_lossとimage_lossの合計をD_lossとして定義'''
            generator_loss.backward()  # 逆伝搬
            optimizerG.step()  # 重み更新
            netG.zero_grad()  # 勾配初期化

            '''それぞれの画像サイズにおけるG_lossの平均を定義'''
            if it == args.log_inter:
                g_loss_val = to_numpy(generator_loss).mean()
                log_dict['G']['all']['g_loss_val'].append(g_loss_val)

            '''訓練状態のlog表示とサンプル生成画像の保存'''
            if (it % args.log_inter) == 0 and it != 0:
                d_loss_val = to_numpy(discriminator_loss).mean()
                g_loss_val = to_numpy(generator_loss).mean()
                print('Epoch: {} [{}/{} ({:.0f}%)]  learnig_rate: {:.4f} g_loss = {:.4f} d_loss= {:.4f}'
                      .format(epoch, it*args.batch_size, updates_per_epoch*args.batch_size,
                              100.*((it*args.batch_size)/(updates_per_epoch*args.batch_size)),
                              g_lr, g_loss_val, d_loss_val))
                logger.debug(
                    'Epoch: {} [{}/{} ({:.0f}%)]  learnig_rate: {:.4f} g_loss = {:.4f} d_loss= {:.4f}'
                    .format(epoch, it*args.batch_size, updates_per_epoch*args.batch_size,
                            100.*((it*args.batch_size)/(updates_per_epoch*args.batch_size)),
                            g_lr, g_loss_val, d_loss_val))

            if it == args.log_inter:
                for k, sample in fake_images.items():
                    save_timgs(images[k],
                               k, 'train_images', path=model_folder, model_name=model_name)
            if it % args.verbose_per_iter == 0:
                for k, sample in fake_images.items():
                    save_imgs(sample,
                              epoch, k, 'train_images', path=model_folder, model_name=model_name)

                """lossの可視化(プロット)"""
            if it == args.log_inter:
                '''dataframe生成'''
                log_df_epoch = pd.DataFrame({'epoch': np.arange(start_epoch, tot_epoch)})


                log_df_D64_pair = pd.DataFrame({'pair_loss_D64': log_dict['D']['output_64']['pair_loss']})
                log_df_D64_image = pd.DataFrame({'image_loss_D64': log_dict['D']['output_64']['image_loss']})

                log_df_D128_pair = pd.DataFrame({'pair_loss_D128': log_dict['D']['output_128']['pair_loss']})
                log_df_D128_image = pd.DataFrame({'image_loss_D128': log_dict['D']['output_128']['image_loss']})

                log_df_D256_pair = pd.DataFrame({'pair_loss_D256': log_dict['D']['output_256']['pair_loss']})
                log_df_D256_image = pd.DataFrame({'image_loss_D256': log_dict['D']['output_256']['image_loss']})

                log_df_D = pd.DataFrame({'D_loss': log_dict['D']['all']['d_loss_val']})


                log_df_G64_pair = pd.DataFrame({'pair_loss_G64': log_dict['G']['output_64']['pair_loss']})
                log_df_G64_image = pd.DataFrame({'image_loss_G64': log_dict['G']['output_64']['image_loss']})

                log_df_G128_pair = pd.DataFrame({'pair_loss_G128': log_dict['G']['output_128']['pair_loss']})
                log_df_G128_image = pd.DataFrame({'image_loss_G128': log_dict['G']['output_128']['image_loss']})

                log_df_G256_pair = pd.DataFrame({'pair_loss_G256': log_dict['G']['output_256']['pair_loss']})
                log_df_G256_image = pd.DataFrame({'image_loss_G256': log_dict['G']['output_256']['image_loss']})

                log_df_G = pd.DataFrame({'G_loss': log_dict['G']['all']['g_loss_val']})


                '''連結'''
                write_df = pd.concat([log_df_epoch, log_df_D, log_df_G,
                                      log_df_D64_pair, log_df_D64_image,
                                      log_df_D128_pair, log_df_D128_image,
                                      log_df_D256_pair, log_df_D256_image,
                                      log_df_G64_pair, log_df_G64_image,
                                      log_df_G128_pair, log_df_G128_image,
                                      log_df_G256_pair, log_df_G256_image,
                                      ], axis=1)

                '''csvファイルに記述'''
                write_df.to_csv(os.path.join(model_root, model_name, '{}.csv'.format('Loss')))
                csv_path = os.path.join(model_root, model_name, '{}.csv'.format('Loss'))

                '''プロット'''
                log_tb = pd.read_csv(csv_path)

                train_D = log_tb['D_loss'].values
                train_G = log_tb['G_loss'].values
                epochs = log_tb['epoch'].values

                fig = plt.figure()
                fig.patch.set_facecolor('white')

                plt.xlabel('epoch')
                plt.ylabel('Loss')
                plt.plot(epochs, train_D, label='D_loss')
                plt.plot(epochs, train_G, label='G_loss')
                plt.title('Training loss ({})'.format(model_name))
                plt.legend()
                if os.path.exists(os.path.join(model_root, model_name, 'Training loss {}.jpg'.format(model_name))):
                    os.remove(os.path.join(model_root, model_name, 'Training loss {}.jpg'.format(model_name)))
                plt.savefig(os.path.join(model_root, model_name, 'Training loss {}.jpg'.format(model_name)))

                plt.close(fig)

                del log_tb

            logcount = 1

        end_timer = time.time() - start_timer

        '''Modelの保存'''
        if epoch % args.save_freq == 0:
            netD = netD.cpu()
            netG = netG.cpu()
            netD_ = netD.module if 'DataParallel' in str(type(netD)) else netD
            netG_ = netG.module if 'DataParallel' in str(type(netD)) else netG
            torch.save(netD_.state_dict(), os.path.join(
                os.path.join(model_root, model_name), 'D_epoch{}.pth'.format(epoch)))
            torch.save(netG_.state_dict(), os.path.join(
                os.path.join(model_root, model_name), 'G_epoch{}.pth'.format(epoch)))
            print('model_saving： {}'.format(os.path.join(model_root, model_name)))
            netD = netD.cuda()
            netG = netG.cuda()
        print(
            'epoch {}/{} is {}minutes...'.format(epoch, tot_epoch, end_timer//60))
