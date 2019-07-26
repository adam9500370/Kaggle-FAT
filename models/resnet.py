import torch
import torch.nn as nn
import math

import torchvision.models as models

from models.utils import *


# https://github.com/Jongchan/attention-module/blob/master/MODELS/model_resnet.py
def conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False):
    "kxk convolution with padding"
    if stride > 1 or kernel_size == 1:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation, groups=groups)
    else:
        return Conv2dDecomposition(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation, groups=groups)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_type='bn', use_cbam=False):
        super(BasicBlock, self).__init__()

        self.conv1 = conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes) if norm_type == 'bn' else nn.GroupNorm(num_groups=32, num_channels=planes)
        self.conv2 = conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes) if norm_type == 'bn' else nn.GroupNorm(num_groups=32, num_channels=planes)
        self.relu = nn.ReLU6(inplace=True)
        self.downsample = downsample

        self.cbam = CBAM( planes, 16 ) if use_cbam else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.cbam is not None:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_type='bn', use_cbam=False):
        super(Bottleneck, self).__init__()

        self.conv1 = conv2d(inplanes, planes, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(planes) if norm_type == 'bn' else nn.GroupNorm(num_groups=32, num_channels=planes)
        self.conv2 = conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes) if norm_type == 'bn' else nn.GroupNorm(num_groups=32, num_channels=planes)
        self.conv3 = conv2d(planes, planes * self.expansion, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if norm_type == 'bn' else nn.GroupNorm(num_groups=32, num_channels=planes * self.expansion)
        self.relu = nn.ReLU6(inplace=True)
        self.downsample = downsample

        self.cbam = CBAM( planes * self.expansion, 16 ) if use_cbam else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.cbam is not None:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


model_dict = {'resnet18':  {'block': BasicBlock, 'layers': [2, 2, 2, 2]},
              'resnet34':  {'block': BasicBlock, 'layers': [3, 4, 6, 3]},
              'resnet50':  {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
              'resnet101': {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
              'resnet152': {'block': Bottleneck, 'layers': [3, 8, 36, 3]},}

class resnet(nn.Module):
    def __init__(self, name='resnet50', n_classes=1000, in_channels=3, norm_type='bn', zero_init_residual=True, use_cbam=False, dropout_rate=0.5):
        super(resnet, self).__init__()

        self.name = name
        self.n_classes = n_classes
        self.use_cbam = use_cbam

        block = model_dict[self.name]['block']
        layers = model_dict[self.name]['layers']

        self.inplanes = 64
        self.dilation = 1
        conv1 = conv2d(in_channels, self.inplanes//2, kernel_size=3, stride=2, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(self.inplanes//2) if norm_type == 'bn' else nn.GroupNorm(num_groups=32, num_channels=self.inplanes//2)
        conv2 = conv2d(self.inplanes//2, self.inplanes//2, kernel_size=3, stride=1, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(self.inplanes//2) if norm_type == 'bn' else nn.GroupNorm(num_groups=32, num_channels=self.inplanes//2)
        conv3 = conv2d(self.inplanes//2, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        bn3 = nn.BatchNorm2d(self.inplanes) if norm_type == 'bn' else nn.GroupNorm(num_groups=32, num_channels=self.inplanes)
        relu = nn.ReLU6(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer0 = nn.Sequential(conv1, bn1, relu, conv2, bn2, relu, conv3, bn3, relu, maxpool)
        self.layer1 = self._make_layer(block, 64,  layers[0], norm_type=norm_type, use_cbam=use_cbam)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_type=norm_type, use_cbam=use_cbam)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_type=norm_type, use_cbam=use_cbam)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_type=norm_type, use_cbam=use_cbam)

        self.fc = nn.Linear(512*block.expansion, self.n_classes, bias=True)

        self.dropout = nn.Dropout(p=dropout_rate)

        self._init_weights(zero_init_residual=zero_init_residual)

    def _init_weights(self, zero_init_residual=True):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    if 'fc' in name:
                        fc_init_bias = - math.log(self.n_classes - 1.)
                        nn.init.constant_(m.bias, fc_init_bias)
                        print('fc sigmoid init bias: {}'.format(fc_init_bias))
                    else:
                        nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for name, m in self.named_modules():
                if self.use_cbam:
                    if isinstance(m, SpatialGate):
                        print('0 weight init for CBAM SpatialGate BN: {}'.format(name))
                        nn.init.constant_(m.spatial[1].weight, 0)
                else:
                    if isinstance(m, Bottleneck):
                        print('0 weight init for Bottleneck BN3: {}'.format(name))
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        print('0 weight init for BasicBlock BN2: {}'.format(name))
                        nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_type='bn', use_cbam=False):
        downsample = None
        norm_layer = nn.BatchNorm2d(planes * block.expansion) if norm_type == 'bn' else nn.GroupNorm(num_groups=32, num_channels=planes * block.expansion)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, padding=0, bias=False),
                norm_layer,
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, norm_type=norm_type, use_cbam=use_cbam))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_type=norm_type, use_cbam=use_cbam))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        gap = F.adaptive_avg_pool2d(x4, output_size=(1, 1)) # Global Average Pooling
        gap = gap.view(gap.size(0), -1) # flatten

        gap = self.dropout(gap)
        out = self.fc(gap)
        return out
