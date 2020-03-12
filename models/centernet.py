# !/usr/bin/python3
# -*- coding: utf-8 -*-
#
# @File    : centernet.py
# @Author  : lk
# @Email   : lk123400@163.com
# @Time    : 2020/1/15 14:16
# Copyright 2020 lk <lk123400@163.com>

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision


class Centernet(nn.Module):
    """
    LPR detectnet
    """
    def __init__(self, modelname):
        super(Centernet, self).__init__()
        if modelname == 'resnet18':
            model = torchvision.models.resnet18()
        elif modelname == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
        print(model)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.conv = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool
        )

        self.layer = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        )

        self.net = nn.Sequential(
            self.conv,
            self.layer
        )

        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.score_map = nn.Conv2d(3840, 1, 1)
        self.center_map = nn.Conv2d(3840, 1, 1)
        self.offset_map = nn.Conv2d(3840, 8, 1)

    def forward(self, x):
        x = self.conv(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        batch_size = x.shape[0]
        """
        print(x3.shape)
        print(x2.shape)
        print(x1.shape)
        """
        x3_up = torch.cat((x3, self.up_sample(x4)), 1)
        x2_up = torch.cat((x2, self.up_sample(x3_up)), 1)
        x1_up = torch.cat((x1, self.up_sample(x2_up)), 1)

        #print(x1_up.shape)
        """
        score_map = torch.sigmoid(self.score_map(x1_up).reshape((batch_size, -1, 2)))
        offset_map = torch.sigmoid(self.offset_map(x1_up).reshape(batch_size, -1, 2))
        center_map = self.center_map(x1_up).reshape(batch_size, -1, 8)
        """
        score_map = torch.sigmoid(self.score_map(x1_up))
        offset_map = torch.sigmoid(self.offset_map(x1_up))
        center_map = self.center_map(x1_up).reshape(batch_size, -1, 8)


        return score_map, offset_map, center_map





