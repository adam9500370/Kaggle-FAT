import os
import cv2
import torch
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF

import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt

from torch.utils import data

from tqdm import tqdm


class freesoundLoader(data.Dataset):
    def __init__(self, root, split="train_curated", mode='wav',
                 is_transform=True, augmentations=None,
                 sampling_rate=44100, duration=2, stft_window_seconds=0.025, stft_hop_seconds=0.010, pad_mode='wrap', pad_onesided=True,
                 no_gt=False, fold_num=0, num_folds=1, seed=1234, train_noisy_loss_weight=0.8, tta=0):
        self.root = root
        self.split = split
        self.mode = mode
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.no_gt = no_gt
        self.n_classes = 80
        self.tta = tta
        self.mean_img = 0.0
        self.std_img = 1.0
        self.train_noisy_loss_weight = train_noisy_loss_weight
        self.files = {}

        # Configs for audio I/O & melspectrogram
        # https://github.com/DCASE-REPO/dcase2019_task2_baseline
        self.sampling_rate = sampling_rate
        self.duration = duration # seconds
        self.samples = self.sampling_rate * self.duration
        self.fmin = 20
        self.fmax = self.sampling_rate // 2
        self.n_mels = 128
        self.stft_window_seconds = stft_window_seconds
        self.window_length = int(np.ceil(self.sampling_rate * self.stft_window_seconds))
        self.n_fft = 2 ** int(np.ceil(np.log2(self.window_length))) # win_length = n_fft
        self.stft_hop_seconds = stft_hop_seconds
        self.hop_length = int(np.ceil(self.sampling_rate * self.stft_hop_seconds))
        self.img_cols = int(np.ceil(float(self.samples - self.n_fft) / self.hop_length)) + 1
        self.pad_mode = pad_mode # ['constant', 'wrap']
        self.pad_onesided = pad_onesided
        print('window_length = {}\nn_fft = {}\nhop_length = {}\nimg_cols = {}'.format(self.window_length, self.n_fft, self.hop_length, self.img_cols))

        test_df = pd.read_csv(os.path.join(self.root, 'sample_submission.csv'), index_col=0)
        self.files['test'] = [os.path.join(self.root, 'test', fname) for fname in list(test_df.index)]
        self.class_names = sorted(list(test_df.columns.values))
        self.class_name2idx = {name: idx for idx, name in enumerate(self.class_names)}

        train_curated_df = pd.read_csv(os.path.join(self.root, 'train_curated.csv'), index_col=0) # train_curated: 4970
        train_curated_rm_fname_list = ['f76181c4.wav', '77b925c2.wav', '6a1f682a.wav', 'c7db12aa.wav', '7752cc8a.wav', '1d44b0bd.wav'] # detected corrupted files in the curated train set
        self.files['train_curated_all'] = [os.path.join(self.root, 'train_curated', fname) for fname in list(train_curated_df.index) if fname not in train_curated_rm_fname_list]
        train_noisy_df = pd.read_csv(os.path.join(self.root, 'train_noisy.csv'), index_col=0) # train_noisy: 19815
        self.files['train_noisy_all'] = [os.path.join(self.root, 'train_noisy', fname) for fname in list(train_noisy_df.index)] + [os.path.join(self.root, 'train_curated', fname) for fname in train_curated_rm_fname_list]

        if self.split != 'test':
            self.train_labels = train_curated_df.to_dict('index')
            self.train_labels.update(train_noisy_df.to_dict('index'))

            if 'noisy' not in self.split: # only using k-folds for 'curated'
                N = len(self.files['train_curated_all'])
                torch.manual_seed(seed)
                rp = torch.randperm(N).tolist()
                start_idx = N * fold_num // num_folds
                end_idx = N * (fold_num + 1) // num_folds
                print('{:5s}: {:2d}/{:2d} [{:6d}, {:6d}] - {:6d}'.format(self.split, fold_num, num_folds, start_idx, end_idx, N))
                self.files[self.split] = []
                for i in range(N):
                    if ((i >= start_idx and i < end_idx) and 'val' in self.split) or (num_folds == 1):
                        self.files[self.split].append(self.files['train_curated_all'][rp[i]])
                    elif (not (i >= start_idx and i < end_idx) and 'train' in self.split):
                        self.files[self.split].append(self.files['train_curated_all'][rp[i]])

                if self.split == 'train': # 'all'
                    self.files[self.split] = self.files[self.split] + self.files['train_noisy_all']
            else: # 'noisy'
                self.files[self.split] = self.files['train_noisy_all']

            self.class_num_samples = torch.zeros(self.n_classes, dtype=torch.float, device=torch.device('cuda'))
            for i, fpath in enumerate(self.files[self.split]):
                fname = os.path.basename(fpath)
                lbl_str = self.train_labels[fname]['labels']
                for l in lbl_str.split(','):
                    self.class_num_samples[self.class_name2idx[l]] += 1
            print('# samples: {:8.1f} ({:6.0f}/{:6.0f})'.format(self.class_num_samples.mean().item(), self.class_num_samples.min().item(), self.class_num_samples.max().item()))

        if not self.files[self.split]:
            raise Exception("No files for split=[%s] found in %s" % (self.split, self.root))
        else:
            print("Found %d %s images" % (len(self.files[self.split]), self.split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        fpath = self.files[self.split][index].rstrip()
        fname = os.path.basename(fpath)

        if self.mode == 'wav':
            img = self.read_as_melspectrogram(fpath, trim_long_data=False, save_npy=True)
            ##img = self.mono_norm(img, to_color=False, to_uint8=False)
            img = np.array(img, dtype=np.float32)
        elif self.mode == 'png':
            img_path = fpath.replace('.wav', '_{}.png'.format(self.sampling_rate))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = np.array(img, dtype=np.uint8)
        else: # npy
            img_path = fpath.replace('.wav', '_{}.npy'.format(self.sampling_rate))
            img = np.load(img_path)
            img = np.array(img, dtype=np.float32)

        if len(img.shape) < 3:
            img = np.expand_dims(img, axis=2) # gray-scale

        if self.augmentations is not None:
            rand_j = np.random.randint(low=0, high=img.shape[1]-self.img_cols+1) if img.shape[1] > self.img_cols else 0
            img = img[:, rand_j:rand_j+self.img_cols, :]

        if not self.no_gt: # one-hot encoding
            lbl_str = self.train_labels[fname]['labels']
            lbl = np.zeros((self.n_classes,), dtype=np.int32)
            for l in lbl_str.split(','):
                lbl[self.class_name2idx[l]] = 1
        else:
            lbl = np.zeros((self.n_classes,), dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        if fpath in self.files['train_noisy_all']:
            lbl = lbl * self.train_noisy_loss_weight + (1. - lbl) * (1. - self.train_noisy_loss_weight) / float(self.n_classes - lbl.sum())

        if self.tta > 1:
            start_indices = [int( (img.shape[2]-self.img_cols)*(float(i)/(float(self.tta)-1.)) + 0.5 ) for i in range(self.tta)]
            img = torch.cat([img[:, :, idx:idx+self.img_cols].unsqueeze(0) for idx in start_indices], dim=0)

        return img, lbl, [fname]

    def transform(self, img, lbl):
        if len(img.shape) < 3:
            img = np.expand_dims(img, axis=2) # gray-scale

        if img.dtype == np.uint8:
            img = img.astype(np.float64) / 255.0 # Rescale images from [0, 255] to [0, 1]
        img = (img - self.mean_img) / self.std_img

        img = img.transpose(2, 0, 1) # NHWC -> NCHW

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl


    def read_audio(self, pathname, trim_long_data=False):
        y, sr = sf.read(pathname, dtype='float32')
        if y.shape[0] > 1:
            y = librosa.to_mono(y)
        if sr != self.sampling_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sampling_rate)

        # trim silence
        if 0 < len(y): # workaround: 0 length causes error
            y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
        # make it unified length to self.samples
        if len(y) > self.samples: # long enough
            if trim_long_data:
                y = y[0:0+self.samples]
        else: # pad blank/replicated
            padding = self.samples - len(y)
            offset = padding // 2 if not self.pad_onesided else 0 # add padding at both ends or not
            y = np.pad(y, (offset, self.samples - len(y) - offset), self.pad_mode)
        return y


    def audio_to_melspectrogram(self, audio, pathname=None):
        pow_spectrogram = librosa.feature.melspectrogram(audio,
                                                         sr=self.sampling_rate,
                                                         n_mels=self.n_mels,
                                                         hop_length=self.hop_length,
                                                         n_fft=self.n_fft,
                                                         fmin=self.fmin,
                                                         fmax=self.fmax,
                                                         power=2)
        log_spectrogram = librosa.power_to_db(pow_spectrogram)
        log_spectrogram = log_spectrogram.astype(np.float32)
        if pathname is not None:
            np.save(pathname.replace('.wav', '_{}.npy'.format(self.sampling_rate)), log_spectrogram)
        return log_spectrogram


    def audio_to_pcen(self, audio, pathname=None): # use power=1 to get a magnitude spectrum instead of a power spectrum
        mag_spectrogram = librosa.feature.melspectrogram(audio,
                                                         sr=self.sampling_rate,
                                                         n_mels=self.n_mels,
                                                         hop_length=self.hop_length,
                                                         n_fft=self.n_fft,
                                                         fmin=self.fmin,
                                                         fmax=self.fmax,
                                                         power=1)
        if pathname is not None:
            np.save(pathname.replace('.wav', '_{}.npy'.format(self.sampling_rate)), mag_spectrogram.astype(np.float32))
            return mag_spectrogram

        # https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion/91859#529792
        pcen_spectrogram = librosa.pcen(mag_spectrogram,
                                        sr=self.sampling_rate,
                                        hop_length=self.hop_length,
                                        gain=0.5,
                                        bias=0.001,
                                        power=0.2,
                                        time_constant=0.4,
                                        eps=1e-9)
        pcen_spectrogram = pcen_spectrogram.astype(np.float32)
        if pathname is not None:
            np.save(pathname.replace('.wav', '_{}.npy'.format(self.sampling_rate)), pcen_spectrogram)
        return pcen_spectrogram


    def show_melspectrogram(self, mels, title='Log-frequency power spectrogram'):
        librosa.display.specshow(mels, x_axis='time', y_axis='mel', 
                                 sr=self.sampling_rate, hop_length=self.hop_length,
                                 fmin=self.fmin, fmax=self.fmax)
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.show()


    def read_as_melspectrogram(self, pathname, trim_long_data=False, debug_display=False, save_npy=False):
        x = self.read_audio(pathname, trim_long_data=trim_long_data)
        mels = self.audio_to_pcen(x, pathname=pathname if save_npy else None)
        if debug_display:
            IPython.display.display(IPython.display.Audio(x, rate=self.sampling_rate))
            self.show_melspectrogram(mels)
        return mels


    def mono_norm(self, X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6, to_color=True, to_uint8=True):
        # Stack X as [X,X,X]
        X = np.stack([X, X, X], axis=-1) if to_color else X

        # Standardize
        mean = mean or X.mean()
        X = X - mean
        std = std or X.std()
        Xstd = X / (std + eps)
        if not to_uint8:
            return Xstd

        _min, _max = Xstd.min(), Xstd.max()
        norm_max = norm_max or _max
        norm_min = norm_min or _min
        if (_max - _min) > eps:
            # Normalize to [0, 255]
            V = Xstd
            V[V < norm_min] = norm_min
            V[V > norm_max] = norm_max
            V = 255 * (V - norm_min) / (norm_max - norm_min)
            V = V.astype(np.uint8)
        else:
            # Just zero
            V = np.zeros_like(Xstd, dtype=np.uint8)
        return V


    def convert_dataset(self, split, trim_long_data=False, save_img=False, save_npy=True, to_color=True, to_uint8=True):
        fpaths = self.files[split]
        for fpath in tqdm(fpaths):
            x = self.read_as_melspectrogram(fpath, trim_long_data=trim_long_data, save_npy=save_npy)
            x_norm = self.mono_norm(x, to_color=to_color, to_uint8=to_uint8)
            if save_img:
                img_path = fpath.replace('.wav', '_{}.png'.format(self.sampling_rate))
                cv2.imwrite(img_path, x_norm)



if __name__ == '__main__':
    local_path = '../datasets/freesound/'

    for sampling_rate in [44100]:
        for split in ["train_curated", "train_noisy"]:
            dst = freesoundLoader(local_path, split=split, fold_num=0, num_folds=1, sampling_rate=sampling_rate, duration=2)
            dst.convert_dataset(split=split, save_img=False, save_npy=True, to_color=False, to_uint8=False)
