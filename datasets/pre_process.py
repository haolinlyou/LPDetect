# !/usr/bin/python3
# -*- coding: utf-8 -*-
#
# @File    : pre_process.py
# @Author  : lk
# @Email   : lk123400@163.com
# @Time    : 2020/1/15 15:40
# Copyright 2020 lk <lk123400@163.com>

import os
import random

with open('train.txt', 'w') as f:
    for filename in os.listdir('lpr_data'):
        line = 'lpr_data/%s %s %s %s %s %s\n'%(filename, random.randint(0,1),
                                               random.randint(120,135), random.randint(50,65),
                                               random.randint(170,195), random.randint(90,125))
        f.writelines(line)
