# !/usr/bin/python3
# -*- coding: utf-8 -*-
#
# @File    : data.py
# @Author  : lk
# @Email   : lk123400@163.com
# @Time    : 2020/1/15 15:17
# Copyright 2020 lk <lk123400@163.com>

from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform, loader=default_loader, input_size=256):
        self.input_size=input_size
        self.transform = transform
        self.loader = loader
        lines = open(txt, 'r')
        data_tag = []
        for line in lines:
            line = line.strip('\n').rstrip()
            words = line.split()
            img_name = words[0]
            label = words[1]
            coor = words[2:]

            data_tag.append((img_name, int(label), np.array(list(map(int, coor)))))
            print(img_name)
            """"""
        self.data_tag = data_tag




    def __getitem__(self, index):
        img_name, label, coor = self.data_tag[index]
        img = self.loader('datasets/' + img_name)
        w, h = img.size
        scale_x, scale_y = self.input_size / w, self.input_size / h
        img = self.transform(img)
        _, center = self.preprocess(coor, scale_x, scale_y)
        heat_map = self.center_gaussian_heatMap(int(self.input_size/4), int(self.input_size/4), center[0], center[1], 2)
        im_ = heat_map*255
        #plt.imshow(im_)
        #plt.show()
        return img, label, heat_map

    def preprocess(self, coor, scale_x, scale_y):
        coor[0::2] = coor[0::2] *scale_x
        coor[1::2] = coor[1::2] *scale_y
        coor = list(map(lambda x:x/4, coor))
        #h, w = coor[3] - coor[1], coor[2] - coor[0]
        cx, cy = np.mean(coor[0::2]), np.mean(coor[1::2])
        off_x1, off_y1 = (cx - coor[0]) / 2, (cy - coor[1]) / 2
        off_x2, off_y2 = (coor[2] - cx) / 2, (coor[3] - cy) / 2
        return (off_x1, off_y1, off_x2, off_y2), (cx, cy)

    def center_label_heatMap(img_width, img_height, c_x, c_y, sigma):
        X1 = np.linspace(1, img_width, img_width)
        Y1 = np.linspace(1, img_height, img_height)
        [X, Y] = np.meshgrid(X1, Y1)
        X = X - c_x
        Y = Y - c_y
        D2 = X * X + Y * Y
        E2 = 2.0 * sigma * sigma
        Exponent = D2 / E2
        heatmap = np.exp(-Exponent)
        return heatmap

    # Compute gaussian kernel
    def center_gaussian_heatMap(self, img_height, img_width, c_x, c_y, variance):
        gaussian_map = np.zeros((img_height, img_width))
        for x_p in range(img_width):
            for y_p in range(img_height):
                dist_sq = (x_p - c_x) * (x_p - c_x) + \
                          (y_p - c_y) * (y_p - c_y)
                exponent = dist_sq / 2.0 / variance / variance
                gaussian_map[y_p, x_p] = np.exp(-exponent)
        return gaussian_map


    def __len__(self):
        return len(self.data_tag)