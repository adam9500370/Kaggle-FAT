# Kaggle-FAT
PyTorch implementation for [Kaggle Freesound Audio Tagging 2019 Challenge](https://www.kaggle.com/c/freesound-audio-tagging-2019)

**36th / 880** (top 5%)

Final leaderboard scores (label-weighted label-ranking average precision (lwlrap)):

0.72340 (private) / 0.715 (public)


## Brief summary
* Using both curated and noisy data for training from scratch
	- Only creating 5 folds for curated data
	- Using all of noisy data for each fold (but using lower confidence weights)
* Raw audio to PCEN [1] (fixed parameters)
* Augmentations
	- Random crop (2 seconds)
	- mixup [2]
	- SpecAugment (frequency mask and time mask) [3]
* Small models
	- ResNet18 with CBAM [4]
	- MobileNetV3-Large [5]
* Losses
	- Binary cross-entropy loss
	- Lovász hinge loss [6]
* TTA: adaptive and overlapped crops (2 seconds) for each raw audio


## Requirements
* pytorch 0.4.1
* torchvision 0.2.0
* librosa
* soundfile
* opencv
* pandas


## Usage

### Data pre-processing
* Modify the dataset path in `config.json`
* Run `PYTHONPATH=. python loaders/freesound_loader.py` to convert to melspectrogram (saved as .npy file)

### To train the model
```
k=5
for((i=0;i<$k;i=i+1))
do 
python train.py --arch resnet18 --dataset freesound --split train \
                --img_rows 128 --img_cols 197 \
                --n_iter 40000 --batch_size 128 --seed 1234 \
                --l_rate 1e-1 --weight_decay 1e-4 --iter_size 1 \
                --num_cycles 0 --print_train_freq 100 --eval_freq 2000 \
                --fold_num $i --num_folds $k --sampling_rate 44100 \
                --dropout_rate 0.5 --gamma_fl 0.0 \
                --use_mix_up --use_spec_aug --use_cbam
done
```
`python train.py -h` for more details

### To test the model
```
k=5
for((i=0;i<$k;i=i+1))
do
python test.py --model_path checkpoints/resnet18_freesound_best_128x197_44100_$i-$k_model.pth --dataset freesound \
               --img_rows 128 --img_cols 197 --seed 1234 \
               --batch_size 32 --split test --sampling_rate 44100 \
               --tta 8 --use_cuda --use_cbam
done
```
`python test.py -h` for more details

### To create final submission
```
python merge.py --dataset freesound --img_rows 128 --img_cols 197 --seed 1234 --split test
```
`python merge.py -h` for more details


## References
[1] [Per-Channel Energy Normalization: Why and How](https://bmcfee.github.io/papers/spl2019_pcen.pdf)

[2] [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

[3] [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html)

[4] [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)

[5] [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

[6] [The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/abs/1705.08790)
