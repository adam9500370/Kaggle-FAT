import sys, os
import cv2
import torch
import argparse
import timeit
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils import data
from tqdm import tqdm

from models import get_model
from loaders import get_loader, get_data_path
from misc.spec_augment import SpecAugment
from misc.losses import *
from misc.lovasz_losses import *
from misc.metrics import *
from misc.scheduler import GradualWarmupScheduler
from misc.utils import convert_state_dict, poly_lr_scheduler, AverageMeter

from torchaudio_contrib.layers import Melspectrogram, Pcen

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm) or isinstance(m, nn.modules.normalization.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        else: # for PCEN
            if hasattr(m, 'log_gain') and  m.log_gain is not None:
                group_no_decay.append(m.log_gain)
            if hasattr(m, 'log_bias') and  m.log_bias is not None:
                group_no_decay.append(m.log_bias)
            if hasattr(m, 'log_power') and  m.log_power is not None:
                group_no_decay.append(m.log_power)
            if hasattr(m, 'log_b') and  m.log_b is not None:
                group_no_decay.append(m.log_b)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def train(args):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # Setup Augmentations
    data_aug = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomCrop(size=(args.img_rows, args.img_cols)),
                ])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, split=args.split, fold_num=args.fold_num, num_folds=args.num_folds, seed=args.seed, augmentations=data_aug, sampling_rate=args.sampling_rate, mode='npy')
    v_loader = data_loader(data_path, is_transform=True, split=args.split.replace('train', 'val'), fold_num=args.fold_num, num_folds=args.num_folds, seed=args.seed, sampling_rate=args.sampling_rate, mode='npy')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
    valloader = data.DataLoader(v_loader, batch_size=1, num_workers=4, pin_memory=True)

    # Setup Model
    model = get_model(args.arch, n_classes, use_cbam=args.use_cbam, in_channels=1, dropout_rate=args.dropout_rate)
    model.cuda()

    """
    mel_spec_layer = Melspectrogram(num_bands=t_loader.n_mels,
                                    sample_rate=t_loader.sampling_rate,
                                    min_freq=t_loader.fmin,
                                    max_freq=t_loader.fmax,
                                    fft_len=t_loader.n_fft,
                                    hop_len=t_loader.hop_length,
                                    power=1.,)
    mel_spec_layer.cuda()
    #"""
    #"""
    # https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/91859#529792
    pcen_layer = Pcen(sr=t_loader.sampling_rate,
                      hop_length=t_loader.hop_length,
                      num_bands=t_loader.n_mels,
                      gain=0.5,
                      bias=0.001,
                      power=0.2,
                      time_constant=0.4,
                      eps=1e-9,
                      trainable=args.pcen_trainable,)
    pcen_layer.cuda()
    #"""

    # Check if model has custom optimizer / loss
    if hasattr(model, 'optimizer'):
        optimizer = model.optimizer
    else:
        warmup_iter = int(args.n_iter*5./100.)
        milestones = [int(args.n_iter*30./100.) - warmup_iter, int(args.n_iter*60./100.) - warmup_iter, int(args.n_iter*90./100.) - warmup_iter] # [30, 60, 90]
        gamma = 0.1

        if args.pcen_trainable:
            optimizer = torch.optim.SGD(group_weight(model) + group_weight(pcen_layer), lr=args.l_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(group_weight(model), lr=args.l_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.num_cycles > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_iter//args.num_cycles, eta_min=args.l_rate*0.01)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        scheduler_warmup = GradualWarmupScheduler(optimizer, total_epoch=warmup_iter, min_lr_mul=0.1, after_scheduler=scheduler)

    start_iter = 0
    if args.resume is not None:                                         
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, encoding="latin1")

            model_dict = model.state_dict()
            if checkpoint.get('model_state', None) is not None:
                model_dict.update(convert_state_dict(checkpoint['model_state'], load_classifier=args.load_classifier))
            else:
                model_dict.update(convert_state_dict(checkpoint, load_classifier=args.load_classifier))
            model.load_state_dict(model_dict)

            if args.pcen_trainable:
                pcen_layer_dict = pcen_layer.state_dict()
                if checkpoint.get('pcen_state', None) is not None:
                    pcen_layer_dict.update(convert_state_dict(checkpoint['pcen_state'], load_classifier=args.load_classifier))
                pcen_layer.load_state_dict(pcen_layer_dict)

            if checkpoint.get('lwlrap', None) is not None:
                start_iter = checkpoint['iter']
                print("Loaded checkpoint '{}' (iter {}, lwlrap {:.5f})"
                      .format(args.resume, checkpoint['iter'], checkpoint['lwlrap']))
            elif checkpoint.get('iter', None) is not None:
                start_iter = checkpoint['iter']
                print("Loaded checkpoint '{}' (iter {})"
                      .format(args.resume, checkpoint['iter']))

            if checkpoint.get('optimizer_state', None) is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state'])

            del model_dict
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    start_iter = args.start_iter if args.start_iter >= 0 else start_iter

    trainloader_iter = iter(trainloader)
    optimizer.zero_grad()
    loss_sum = 0.0
    spec_augment = SpecAugment(time_warp_rate=0.1, freq_mask_rate=0.2, time_mask_rate=0.2, num_masks=2) if args.use_spec_aug else None
    best_lwlrap = 0.0
    start_train_time = timeit.default_timer()
    for i in range(start_iter, args.n_iter):
        model.train()
        ##mel_spec_layer.train()
        pcen_layer.train()

        if args.num_cycles == 0:
            scheduler_warmup.step(i)
        else:
            scheduler_warmup.step(i // args.num_cycles)

        try:
            images, labels, _ = next(trainloader_iter)
        except:
            trainloader_iter = iter(trainloader)
            images, labels, _ = next(trainloader_iter)

        images = images.cuda()
        labels = labels.cuda()

        ##images = mel_spec_layer(images)
        images = pcen_layer(images)

        if args.use_mix_up:
            beta_ab = 0.4
            mix_up_alpha = np.random.beta(size=labels.size(0), a=beta_ab, b=beta_ab)
            mix_up_alpha = np.maximum(mix_up_alpha, 1. - mix_up_alpha)
            mix_up_alpha = torch.from_numpy(mix_up_alpha).float().cuda()

            rand_indices = np.arange(labels.size(0))
            np.random.shuffle(rand_indices)
            rand_indices = torch.from_numpy(rand_indices).long().cuda()
            images2 = torch.index_select(images, dim=0, index=rand_indices)
            labels2 = torch.index_select(labels, dim=0, index=rand_indices)

            images = images * mix_up_alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3) + images2 * (1. -  mix_up_alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3))
            labels = labels * mix_up_alpha.unsqueeze(1) + labels2 * (1. -  mix_up_alpha.unsqueeze(1))

        if args.use_spec_aug:
            images = spec_augment(images, augs=['freq_mask', 'time_mask'])

        outputs = model(images)

        focal_loss = sigmoid_focal_loss_with_logits(outputs, labels, gamma=args.gamma_fl)
        lovasz_loss = lovasz_hinge(outputs, labels)
        loss = focal_loss + lovasz_loss
        loss = loss / float(args.iter_size)
        loss.backward()
        loss_sum = loss_sum + loss.item()

        if (i+1) % args.print_train_freq == 0:
            print("Iter [%7d/%7d] Loss: %7.4f" % (i+1, args.n_iter, loss_sum))

        if (i+1) % args.iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_sum = 0.0

        if args.eval_freq > 0 and (i+1) % (args.eval_freq // args.save_freq) == 0:
            state = {'iter': i+1,
                     'model_state': model.state_dict(),}
                     #'optimizer_state': optimizer.state_dict(),}
            if args.pcen_trainable:
                state['pcen_state'] = pcen_layer.state_dict()
            torch.save(state, "checkpoints/{}_{}_{}_{}x{}_{}_{}-{}_model.pth".format(args.arch, args.dataset, i+1, args.img_rows, args.img_cols, args.sampling_rate, args.fold_num, args.num_folds))

        if args.eval_freq > 0 and (i+1) % args.eval_freq == 0:
            y_true = np.zeros((v_loader.__len__(), n_classes), dtype=np.int32)
            y_prob = np.zeros((v_loader.__len__(), n_classes), dtype=np.float32)
            mean_loss_val = AverageMeter()
            model.eval()
            ##mel_spec_layer.eval()
            pcen_layer.eval()
            with torch.no_grad():
                for i_val, (images_val, labels_val, _) in tqdm(enumerate(valloader)):
                    images_val = images_val.cuda()
                    labels_val = labels_val.cuda()

                    ##images_val = mel_spec_layer(images_val)
                    images_val = pcen_layer(images_val)

                    if images_val.size(-1) > args.img_cols: # split into overlapped chunks
                        stride = (args.img_cols // args.img_cols_div) if (images_val.size(-1) - args.img_cols) > (args.img_cols // args.img_cols_div) else (images_val.size(-1) - args.img_cols)
                        images_val = torch.cat([images_val[:, :, :, w:w+args.img_cols] for w in range(0, images_val.size(-1)-args.img_cols+1, stride)], dim=0)

                    outputs_val = model(images_val)

                    prob_val = F.sigmoid(outputs_val)
                    outputs_val = outputs_val.mean(0, keepdim=True)
                    prob_val = prob_val.mean(0, keepdim=True)

                    focal_loss_val = sigmoid_focal_loss_with_logits(outputs_val, labels_val, gamma=args.gamma_fl)
                    lovasz_loss_val = lovasz_hinge(outputs_val, labels_val)
                    loss_val = focal_loss_val + lovasz_loss_val
                    mean_loss_val.update(loss_val, n=labels_val.size(0))

                    y_true[i_val:i_val+labels_val.size(0), :] = labels_val.long().cpu().numpy()
                    y_prob[i_val:i_val+labels_val.size(0), :] = prob_val.cpu().numpy()

            per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(y_true, y_prob)
            lwlrap_val = np.sum(per_class_lwlrap * weight_per_class)
            print('lwlrap: {:.5f}'.format(lwlrap_val))
            print('Mean val loss: {:.4f}'.format(mean_loss_val.avg))
            state['lwlrap'] = lwlrap_val
            mean_loss_val.reset()
            if (i+1) == args.n_iter:
                print('per_class_lwlrap: {:.5f} ~ {:.5f}'.format(per_class_lwlrap.min(), per_class_lwlrap.max()))
                for c in range(n_classes):
                    print('{:50s}: {:.5f} ({:.5f})'.format(v_loader.class_names[c], per_class_lwlrap[c], weight_per_class[c]))

            torch.save(state, "checkpoints/{}_{}_{}_{}x{}_{}_{}-{}_model.pth".format(args.arch, args.dataset, i+1, args.img_rows, args.img_cols, args.sampling_rate, args.fold_num, args.num_folds))
            if best_lwlrap <= lwlrap_val:
                best_lwlrap = lwlrap_val
                torch.save(state, "checkpoints/{}_{}_{}_{}x{}_{}_{}-{}_model.pth".format(args.arch, args.dataset, 'best', args.img_rows, args.img_cols, args.sampling_rate, args.fold_num, args.num_folds))

            print('-- PCEN --\n gain = {:.5f}/{:.5f}\n bias = {:.5f}/{:.5f}\n power = {:.5f}/{:.5f}\n b = {:.5f}/{:.5f}'.format(
                        pcen_layer.log_gain.exp().min().item(), pcen_layer.log_gain.exp().max().item(),
                        pcen_layer.log_bias.exp().min().item(), pcen_layer.log_bias.exp().max().item(),
                        pcen_layer.log_power.exp().min().item(), pcen_layer.log_power.exp().max().item(),
                        pcen_layer.log_b.exp().min().item(), pcen_layer.log_b.exp().max().item()))

            elapsed_train_time = timeit.default_timer() - start_train_time
            print('Training time (iter {0:5d}): {1:10.5f} seconds'.format(i+1, elapsed_train_time))
            start_train_time = timeit.default_timer()

    print('best_lwlrap: {:.5f}'.format(best_lwlrap))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet18',
                        help='Architecture to use [\'resnet, MobileNetV3, etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='freesound',
                        help='Dataset to use')
    parser.add_argument('--img_rows', nargs='?', type=int, default=128,
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=197,
                        help='Width of the input image')

    parser.add_argument('--split', nargs='?', type=str, default='train',
                        help='Split of dataset to train on')
    parser.add_argument('--n_iter', nargs='?', type=int, default=40000,
                        help='# of the iters')
    parser.add_argument('--batch_size', nargs='?', type=int, default=128,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-1,
                        help='Learning Rate')
    parser.add_argument('--momentum', nargs='?', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--weight_decay', nargs='?', type=float, default=1e-4,
                        help='Weight Decay')
    parser.add_argument('--iter_size', nargs='?', type=int, default=1,
                        help='Accumulated batch gradient size')
    parser.add_argument('--resume', nargs='?', type=str, default=None,   
                        help='Path to previous saved model to restart from')

    parser.add_argument('--load_classifier', dest='load_classifier', action='store_true',
                        help='Enable to load classifier weights | True by default')
    parser.add_argument('--no-load_classifier', dest='load_classifier', action='store_false',
                        help='Disable to load classifier weights | True by default')
    parser.set_defaults(load_classifier=True)

    parser.add_argument('--use_cbam', dest='use_cbam', action='store_true',
                        help='Enable to use CBAM | False by default')
    parser.add_argument('--no-use_cbam', dest='use_cbam', action='store_false',
                        help='Disable to use CBAM | False by default')
    parser.set_defaults(use_cbam=False)

    parser.add_argument('--use_mix_up', dest='use_mix_up', action='store_true',
                        help='Enable to use mix-up | False by default')
    parser.add_argument('--no-use_mix_up', dest='use_mix_up', action='store_false',
                        help='Disable to use mix-up | False by default')
    parser.set_defaults(use_mix_up=False)

    parser.add_argument('--use_spec_aug', dest='use_spec_aug', action='store_true',
                        help='Enable to use SpecAugment | False by default')
    parser.add_argument('--no-use_spec_aug', dest='use_spec_aug', action='store_false',
                        help='Disable to use SpecAugment | False by default')
    parser.set_defaults(use_spec_aug=False)

    parser.add_argument('--pcen_trainable', dest='pcen_trainable', action='store_true',
                        help='Enable to make PCEN trainable | False by default')
    parser.add_argument('--no-pcen_trainable', dest='pcen_trainable', action='store_false',
                        help='Disable to make PCEN trainable | False by default')
    parser.set_defaults(pcen_trainable=False)

    parser.add_argument('--seed', nargs='?', type=int, default=1234, 
                        help='Random seed')
    parser.add_argument('--num_cycles', nargs='?', type=int, default=0, 
                        help='Cosine Annealing Cyclic LR')

    parser.add_argument('--fold_num', nargs='?', type=int, default=0,
                        help='Fold number in each class for training')
    parser.add_argument('--num_folds', nargs='?', type=int, default=5,
                        help='Number of folds for training')
    parser.add_argument('--print_train_freq', nargs='?', type=int, default=100,
                        help='Frequency (iterations) of training logs display')
    parser.add_argument('--eval_freq', nargs='?', type=int, default=2000,
                        help='Frequency (iters) of evaluation of current model')
    parser.add_argument('--save_freq', nargs='?', type=int, default=1,
                        help='Frequency (iters) of saving current model (divided by eval_freq)')

    parser.add_argument('--dropout_rate', nargs='?', type=float, default=0.5,
                        help='Dropout value')

    parser.add_argument('--gamma_fl', nargs='?', type=float, default=0.0,
                        help='Focal Loss - gamma')

    parser.add_argument('--sampling_rate', nargs='?', type=int, default=44100, 
                        help='Audio sampling rate')

    parser.add_argument('--img_cols_div', nargs='?', type=int, default=2,
                        help='Overlapped chunk size for TTA')

    parser.add_argument('--start_iter', nargs='?', type=int, default=-1,
                        help='Starting iteration number (-1 to ignore)')

    args = parser.parse_args()
    print(args)
    train(args)
