import sys, os
import cv2
import torch
import argparse
import timeit
import random
import collections
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils import data
from tqdm import tqdm

from models import get_model
from loaders import get_loader, get_data_path
from misc.metrics import *
from misc.utils import convert_state_dict, AverageMeter

from torchaudio_contrib.layers import Melspectrogram, Pcen

def test(args):
    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[:model_file_name.find('_')]

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True, no_gt=args.no_gt, seed=args.seed, sampling_rate=args.sampling_rate, tta=args.tta)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)

    n_classes = loader.n_classes
    testloader = data.DataLoader(loader, batch_size=args.batch_size, num_workers=1, pin_memory=True)

    # Setup Model
    model = get_model(model_name, n_classes, use_cbam=args.use_cbam, in_channels=1)
    if args.use_cuda:
        model.cuda()

    """
    mel_spec_layer = Melspectrogram(num_bands=loader.n_mels,
                                    sample_rate=loader.sampling_rate,
                                    min_freq=loader.fmin,
                                    max_freq=loader.fmax,
                                    fft_len=loader.n_fft,
                                    hop_len=loader.hop_length,
                                    power=1.,)
    if args.use_cuda:
        mel_spec_layer.cuda()
    #"""
    #"""
    # https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/91859#529792
    pcen_layer = Pcen(sr=loader.sampling_rate,
                      hop_length=loader.hop_length,
                      num_bands=loader.n_mels,
                      gain=0.5,
                      bias=0.001,
                      power=0.2,
                      time_constant=0.4,
                      eps=1e-9,
                      trainable=args.pcen_trainable,)
    if args.use_cuda:
        pcen_layer.cuda()
    #"""

    checkpoint = torch.load(args.model_path, map_location=None if args.use_cuda else 'cpu', encoding="latin1")
    state = convert_state_dict(checkpoint['model_state'])
    model_dict = model.state_dict()
    model_dict.update(state)
    model.load_state_dict(model_dict)
    if args.pcen_trainable:
        pcen_state = convert_state_dict(checkpoint['pcen_state'])
        pcen_layer_dict = pcen_layer.state_dict()
        pcen_layer_dict.update(pcen_state)
        pcen_layer.load_state_dict(pcen_layer_dict)

        print('-- PCEN --\n gain = {:.5f}/{:.5f}\n bias = {:.5f}/{:.5f}\n power = {:.5f}/{:.5f}\n b = {:.5f}/{:.5f}'.format(
                    pcen_layer.log_gain.exp().min().item(), pcen_layer.log_gain.exp().max().item(),
                    pcen_layer.log_bias.exp().min().item(), pcen_layer.log_bias.exp().max().item(),
                    pcen_layer.log_power.exp().min().item(), pcen_layer.log_power.exp().max().item(),
                    pcen_layer.log_b.exp().min().item(), pcen_layer.log_b.exp().max().item()))

    if checkpoint.get('lwlrap', None) is not None:
        print("Loaded checkpoint '{}' (iter {}, lwlrap {:.5f})"
              .format(args.model_path, checkpoint['iter'], checkpoint['lwlrap']))
    else:
        print("Loaded checkpoint '{}' (iter {})"
              .format(args.model_path, checkpoint['iter']))

    y_true = np.zeros((loader.__len__(), n_classes), dtype=np.int32)
    y_prob = np.zeros((loader.__len__(), n_classes), dtype=np.float32)
    model.eval()
    ##mel_spec_layer.eval()
    pcen_layer.eval()
    with torch.no_grad():
        for i, (images, labels, names) in tqdm(enumerate(testloader)):
            if args.use_cuda:
                images = images.cuda()

            if args.tta > 1:
                bs, num_tta, c, h, w = images.size()
                images = images.view(-1, c, h, w)

            ##images = mel_spec_layer(images)
            images = pcen_layer(images)

            outputs = model(images)

            prob = F.sigmoid(outputs)
            if args.tta > 1:
                prob = prob.view(bs, num_tta, -1)
                prob = prob.mean(1)

            if not args.no_gt:
                y_true[i*args.batch_size:i*args.batch_size+labels.size(0), :] = labels.long().cpu().numpy() if args.use_cuda else labels.long().numpy()
            y_prob[i*args.batch_size:i*args.batch_size+labels.size(0), :] = prob.cpu().numpy() if args.use_cuda else prob.numpy()

    n_iter = model_file_name.split('_')[2]
    fold_num, num_folds = model_file_name.split('_')[-2].split('-')
    prob_file_name = '{}_{}x{}_{}_{}_{}_{}-{}'.format(args.split, args.img_rows, args.img_cols, model_name, n_iter, args.sampling_rate, fold_num, num_folds)
    np.save('prob-{}.npy'.format(prob_file_name), y_prob)

    if not args.no_gt:
        lwlrap_val = calculate_overall_lwlrap_sklearn(y_true, y_prob)
        print('lwlrap: {:.5f}'.format(lwlrap_val))

    # Create submission
    csv_file_name = '{}_{}x{}_{}_{}_{}_{}-{}.csv'.format(args.split, args.img_rows, args.img_cols, model_name, n_iter, args.sampling_rate, fold_num, num_folds)
    sub = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'), index_col=0)
    sub[loader.class_names] = y_prob
    sub.to_csv(csv_file_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='resnet18_freesound_best_0-5_model.pth',
                        help='Path to the saved model')
    parser.add_argument('--dataset', nargs='?', type=str, default='freesound',
                        help='Dataset to use')
    parser.add_argument('--img_rows', nargs='?', type=int, default=128,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=197,
                        help='Width of the input image')

    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--split', nargs='?', type=str, default='test',
                        help='Split of dataset to test on')

    parser.add_argument('--use_cbam', dest='use_cbam', action='store_true',
                        help='Enable to use CBAM | False by default')
    parser.add_argument('--no-use_cbam', dest='use_cbam', action='store_false',
                        help='Disable to use CBAM | False by default')
    parser.set_defaults(use_cbam=False)

    parser.add_argument('--pcen_trainable', dest='pcen_trainable', action='store_true',
                        help='Enable to make PCEN trainable | False by default')
    parser.add_argument('--no-pcen_trainable', dest='pcen_trainable', action='store_false',
                        help='Disable to make PCEN trainable | False by default')
    parser.set_defaults(pcen_trainable=False)

    parser.add_argument('--no_gt', dest='no_gt', action='store_true',
                        help='Disable verification | True by default')
    parser.add_argument('--gt', dest='no_gt', action='store_false',
                        help='Enable verification | True by default')
    parser.set_defaults(no_gt=True)

    parser.add_argument('--seed', nargs='?', type=int, default=1234,
                        help='Random seed')

    parser.add_argument('--sampling_rate', nargs='?', type=int, default=44100, 
                        help='Audio sampling rate')

    parser.add_argument('--tta', nargs='?', type=int, default=0,
                        help='# TTA')

    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help='Enable CUDA | True by default')
    parser.add_argument('--no-use_cuda', dest='use_cuda', action='store_false',
                        help='Disable CUDA | True by default')
    parser.set_defaults(use_cuda=True)

    args = parser.parse_args()
    print(args)

    if args.use_cuda:
        from torch.backends import cudnn
        cudnn.benchmark = True
        cudnn.deterministic = True

    test(args)
