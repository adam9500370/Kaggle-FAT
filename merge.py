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

from torch.backends import cudnn
from torch.utils import data
from tqdm import tqdm

from models import get_model
from loaders import get_loader, get_data_path
from misc.utils import convert_state_dict, flip, AverageMeter

import glob

def merge(args):
    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, split=args.split, is_transform=True, no_gt=args.no_gt, seed=args.seed)

    n_classes = loader.n_classes

    avg_y_prob = np.zeros((loader.__len__(), n_classes), dtype=np.float32)
    fold_list = []
    for prob_file_name in glob.glob('*.npy'):
        prob = np.load(prob_file_name)
        avg_y_prob = avg_y_prob + prob
        fold_list.append(prob_file_name)
    avg_y_prob = avg_y_prob / len(fold_list)
    avgprob_file_name = 'prob_{}_avg.npy'.format(len(fold_list))
    np.save(avgprob_file_name, avg_y_prob)

    # Create submission
    csv_file_name = 'submission.csv'
    sub = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'), index_col=0)
    sub[loader.class_names] = avg_y_prob
    sub.to_csv(csv_file_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, default='freesound',
                        help='Dataset to use')
    parser.add_argument('--img_rows', nargs='?', type=int, default=128,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=197,
                        help='Width of the input image')

    parser.add_argument('--split', nargs='?', type=str, default='test',
                        help='Split of dataset to test on')

    parser.add_argument('--no_gt', dest='no_gt', action='store_true',
                        help='Disable verification | True by default')
    parser.add_argument('--gt', dest='no_gt', action='store_false',
                        help='Enable verification | True by default')
    parser.set_defaults(no_gt=True)

    parser.add_argument('--seed', nargs='?', type=int, default=1234,
                        help='Random seed')

    args = parser.parse_args()
    print(args)
    merge(args)
