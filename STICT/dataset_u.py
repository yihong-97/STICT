#!/usr/bin/python3
# coding=utf-8

import os
import random
import torch
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL.Image as Image


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, data_path, frame_dict, clips):
        super().__init__()

        self.frame_dict = frame_dict
        self.clips = clips
        self.datapath = data_path

        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.trans_f = transforms.Compose([
            transforms.ToTensor()
        ])
        # print('unlabel num:', len(self.clips))

    def __getitem__(self, idx):
        clip = self.clips[idx]
        names = clip['clip_frame_index']
        name0 = self.frame_dict[names[0]]

        if 'DS_U' in self.datapath:
            image1_ = Image.open(self.datapath + '/images/' + name0 + '.png').convert('RGB').resize((352, 352))
        else:
            image1_ = Image.open(self.datapath + '/images/' + name0).convert('RGB').resize((352, 352))

        name1 = self.frame_dict[names[1]]

        if 'DS_U' in self.datapath:
            image2_ = Image.open(self.datapath + '/images/' + name1 + '.png').convert('RGB').resize((352, 352))
        else:
            image2_ = Image.open(self.datapath + '/images/' + name1).convert('RGB').resize((352, 352))

        name2 = self.frame_dict[names[2]]

        if 'DS_U' in self.datapath:
            image3_ = Image.open(self.datapath + '/images/' + name2 + '.png').convert('RGB').resize((352, 352))
        else:
            image3_ = Image.open(self.datapath + '/images/' + name2).convert('RGB').resize((352, 352))

        image_nom1 = self.trans(image1_)
        image_nom2 = self.trans(image2_)
        image_nom3 = self.trans(image3_)

        image1_f = self.trans_f(image1_)
        image2_f = self.trans_f(image2_)
        image3_f = self.trans_f(image3_)

        return image_nom1, image_nom2, image_nom3, image1_f, image2_f, image3_f, clip

    def __len__(self):
        return len(self.clips)
