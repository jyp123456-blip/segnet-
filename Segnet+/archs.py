from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from attition.Attition import CBAM
from model.backbone import build_backbone
from add1.MOGA import MogaBlock
from add1.MASAG import MASAG
import numpy as np
from add1.CAS import CAS

__all__ = ['SegNet']
torch.autograd.set_detect_anomaly(True)

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            #ScConv(64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,64, kernel_size=3, padding=1, bias=False),
            #WTConv2d(64, 64, kernel_size=3, bias=False),
            #ScConv(64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.masag1 = MASAG(dim=64)  # 添加注意力模块

        self.encode2 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.masag2 = MASAG(dim=128)  # 添加注意力模块

        self.encode3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.masag3 = MASAG(dim=256)  # 添加注意力模块
        self.encode4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.masag4 = MASAG(dim=512)  # 添加注意力模块

        self.encode5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.masag5 = MASAG(dim=512)  # 添加注意力模块

    def forward(self, x):
        idx = []

        # (3, 512, 512) -> (64, 256, 256)
        x = self.encode1(x)
        x = self.masag1(x, x)  # 添加注意力模块

        x, id1 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id1)

        # (64, 256, 256) -> (128, 128, 128)
        x = self.encode2(x)
        x = self.masag2(x, x)  # 添加注意力模块
        x, id2 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id2)

        # (128, 128, 128) -> (256, 64, 64)
        x = self.encode3(x)
        x = self.masag3(x, x)  # 添加注意力模块

        x, id3 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id3)

        # (256, 64, 64) -> (512, 32, 32)
        x = self.encode4(x)
        x = self.masag4(x, x)  # 添加注意力模块

        x, id4 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id4)

        # (512, 32, 32) -> (512, 16, 16)
        x = self.encode5(x)
        x = self.masag5(x, x)  # 添加注意力模块

        x, id5 = F.max_pool2d_with_indices(x, kernel_size=2, stride=2, return_indices=True)
        idx.append(id5)

        return x, idx

class Deocder(nn.Module):
    def __init__(self, out_channels):
        super(Deocder, self).__init__()

        self.decode1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.moga_block1 = MogaBlock(embed_dims=512)  # 添加注意力模块
        self.decode2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.moga_block2 = MogaBlock(embed_dims=256)  # 添加注意力模块
        self.decode3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.moga_block3 = MogaBlock(embed_dims=128)  # 添加注意力模块
        self.decode4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            #ScConv(128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            #ScConv(64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.moga_block4 = MogaBlock(embed_dims=64)  # 添加注意力模块
        self.decode5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            #ScConv(64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
            #RepConv(64,out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.moga_block5 = MogaBlock(embed_dims=64)  # 添加注意力模块

    def forward(self, x, idx):
        """
        :param x: 经过卷积操作后的特征图
        :param idx: decode中每次最大池化时最大值的位置索引
        """

        # (512, 16, 16) -> (512, 32, 32) -> (512, 32, 32)
        x = F.max_unpool2d(x, idx[4], kernel_size=2, stride=2)
        x = self.decode1(x)
        x = self.moga_block1(x)  # 添加注意力模块


        # (512, 32, 32) -> (512, 64, 64) -> (256, 64, 64)
        x = F.max_unpool2d(x, idx[3], kernel_size=2, stride=2)
        x = self.decode2(x)
        x = self.moga_block2(x)  # 添加注意力模块

        # (256, 64, 64) -> (256, 128, 128) -> (128, 128, 128)
        x = F.max_unpool2d(x, idx[2], kernel_size=2, stride=2)
        x = self.decode3(x)
        x = self.moga_block3(x)  # 添加注意力模块


        # (128, 128, 128) -> (128, 256, 256) -> (64, 256, 256)
        x = F.max_unpool2d(x, idx[1], kernel_size=2, stride=2)
        x = self.decode4(x)
        x = self.moga_block4(x)  # 添加注意力模块


        # (64, 256, 256) -> (64, 512, 512) -> (num_classes, 512, 512)
        x = F.max_unpool2d(x, idx[0], kernel_size=2, stride=2)
        x = self.decode5(x)

        return x


class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()

        self.encoder = Encoder(in_channels=3)
        self.cas = CAS(dim=512)#添加注意力模块
        self.decoder = Deocder(out_channels=num_classes)

    def forward(self, x):
        x, idx = self.encoder(x)
        x = self.cas(x)#添加注意力模块
        x = self.decoder(x, idx)
        return x


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


    