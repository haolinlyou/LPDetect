# !/usr/bin/python3
# -*- coding: utf-8 -*-
#
# @File    : inference.py
# @Author  : lk
# @Email   : lk123400@163.com
# @Time    : 2020/1/16 17:09
# Copyright 2020 lk <lk123400@163.com>

import torch
from models.centernet import Centernet
from torchvision import transforms as transforms
from torch.nn import  functional as F
from PIL import Image
import numpy as np

def default_loader(path):
    return Image.open(path).convert('RGB')

class Inference():
    def __init__(self):
        self.model = Centernet('resnet50')
        pre_dict = torch.load('weights/epoch_19.pth')
        self.model.load_state_dict(pre_dict)
        self.model.train(False)
        self.input_size=128
        self.stride = 4
        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )])

    def predict(self, img):
        w, h = img.size
        print(w, h)
        scale_w, scale_h = self.input_size/w, self.input_size/h
        self.img = self.transform(img)
        self.img = self.img.unsqueeze(0)
        score_map, _, _ = self.model(self.img)
        print(score_map, torch.max(score_map))
        indices = torch.where(score_map == torch.max(score_map))
        print(indices)
        #cx, cy = x*4/scale_w, y*4/scale_h
        #print(cx, cy)

def test(path):
    img = default_loader(path)
    inference = Inference()
    inference.predict(img)


if __name__ == '__main__':
    test('datasets/lpr_data/06_highway_cutin_20s01.jpg')
