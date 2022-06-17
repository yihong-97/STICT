#!/usr/bin/python3
#coding=utf-8

import os
import random
import torch
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL.Image as Image

########################### Data Augmentation ###########################
class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class RandomCrop(object):
    def __call__(self, image, mask):
        W,H   = image.size
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image.crop((p0,p2,p3,p2)), mask.crop((p0,p2,p3,p2))


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, data_path, txt_path):
        super().__init__()
        self.datapath = data_path
        self.hflip = RandomHorizontallyFlip()
        self.crop = RandomCrop()
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with open(txt_path, 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

        print('train num:', len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name  = self.samples[idx]
       
        if  'DS' in self.datapath: 
            image = Image.open(self.datapath+'/images/'+name+'.png').convert('RGB')
        else:
            image = Image.open(self.datapath+'/images/'+name+'.jpg').convert('RGB')
        label  = Image.open(self.datapath+'/labels/' +name+'.png').convert('L')

        return image,label

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image, mask = [list(item) for item in zip(*batch)]

        for i in range(len(batch)):
            image[i] = image[i].resize((size, size))
            mask[i]  =  mask[i].resize((size, size))
            image[i], mask[i] = self.hflip(image[i], mask[i])
            mask[i] = np.array(mask[i], dtype='float32') / 255.0
            if len(mask[i].shape) > 2:
                mask[i] = mask[i][:, :, 0]
            image[i] = self.trans(image[i])
        image = torch.from_numpy(np.stack(image, axis=0))
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
          
        return image, mask


class TestDataset(Dataset):
    def __init__(self, data_path,txt_path):
        super().__init__()
        self.datapath = data_path
        with open(txt_path, 'r') as lines:
            self.imgs = []
            for line in lines:
                self.imgs.append(line.strip())

        self.file_num = len(self.imgs)

        self.hflip = RandomHorizontallyFlip()
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print('test num:', self.file_num)

    def __len__(self):
        return self.file_num

    def __getitem__(self, index):
        name = self.imgs[index]
        if  'DS' in self.datapath: 
            image = Image.open(self.datapath+'/images/'+name+'.png').convert('RGB').resize((352, 352))
        else:
            image = Image.open(self.datapath+'/images/'+name+'.jpg').convert('RGB').resize((352, 352))
        label = Image.open(self.datapath+'/labels/'+name+'.png').convert('L').resize((352, 352))
        shape = np.array(label).shape

        label = np.array(label, dtype='float32') / 255.0
        if len(label.shape) > 2:
            label = label[:, :, 0]

        image_nom = self.trans(image)

        return image_nom,label,name,shape