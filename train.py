# !/usr/bin/python3
# -*- coding: utf-8 -*-
#
# @File    : train.py
# @Author  : lk
# @Email   : lk123400@163.com
# @Time    : 2020/1/15 14:38
# Copyright 2020 lk <lk123400@163.com>

import time
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from models.centernet import Centernet
from datasets.data import MyDataset
import matplotlib.pyplot as plt


batch_size = 2
epoch = 120
use_gpu = False

in_put_size = 256


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def train():
    model = Centernet('resnet50')
    if use_gpu:
        model = model.cuda()
    transform = transforms.Compose([transforms.Resize((in_put_size, in_put_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=(0., 0., 0.),
                                        std=(1.0, 1.0, 1.0)
                                    )])
    lpr_datasets = MyDataset('datasets/train.txt', transform)
    trainloader = torch.utils.data.DataLoader(lpr_datasets, batch_size=batch_size, shuffle=True, num_workers=8)


    optimer = optim.SGD([{'params': model.parameters(), 'lr':0.001}])
    #optimer = optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimer, [80, 90, 100, 120])
    smooth = torch.nn.SmoothL1Loss()
    loss_score = torch.nn.MSELoss()
    ladmark_loss = torch.nn.L1Loss()

    for i in range(epoch):
        scheduler.step()
        model.train(True)
        for j, (data, label, heat_map) in enumerate(trainloader):
            optimer.zero_grad()
            heat_map = torch.tensor(heat_map).float()
            if use_gpu:
                data, label, heat_map = data.cuda(), label.cuda(), heat_map.cuda()
            score_map, offset_map, _ = model(data)
            show_fig(score_map, heat_map)
            loss = loss_score(score_map.reshape(-1, 64, 64), heat_map)


            #loss = classifer(score_map[:,:,1].reshape(batch_size,-1), heat_map.reshape(batch_size, -1))
            print("%s Train Epoch[%s] Step:%s Loss:%.4f Lr:%s"%(get_time(), i, j, loss.cpu().data.item(), scheduler.get_lr()[0]))
            loss.backward()
            """
            for name, parms in model.named_parameters():
                if name == 'conv1.weight':
                    print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                      ' -->grad_value:', parms.grad)
                      """
            optimer.step()
        model_save_name = 'epoch_%s.pth'%(i)
        torch.save(model.state_dict(), './weights/%s'%model_save_name)
        print('Saved model:%s'%model_save_name)


def show_fig(score_map, heatmap):
    #print(heatmap.shape)
    heatmap = heatmap[0, :, :].detach().cpu().numpy()*255
    score_map = score_map[0, 0, :, :].detach().cpu().numpy()*255
    plt.subplot(1,2,1)
    plt.imshow(score_map)
    plt.subplot(1,2,2)
    plt.imshow(heatmap)
    plt.draw()
    plt.pause(0.01)
    #plt.show()






if __name__ == '__main__':
    train()





