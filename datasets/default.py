from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import pandas as pd
import scipy.misc as misc
from torch.utils.data.dataset import Dataset

import torch
from torchvision import models, transforms as T

from PIL import Image

# TODO:
# 1. how to normalize
# 2. how to account for controls

class RCICDefaultDataset(Dataset):
    def __init__(self, df, img_dir, tsfm=None, mode='train', site=1, channels=[1,2,3,4,5,6]):        
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.tsfm = tsfm
        
    def _load_img_as_tensor(self, file_name):
        img = Image.open(file_name)                
        if self.tsfm: img = self.tsfm(img)
        return img

    def _get_img_path(self, index, channel):
        experiment, plate, well = self.records[index].experiment, self.records[index].plate, self.records[index].well
        return '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site}_w{channel}.png'])
        
    def __getitem__(self, index):
        paths = [self._get_img_path(index, ch) for ch in self.channels]        
        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])
        
        if self.mode == 'train':
            return img, self.records[index].id_code, int(self.records[index].sirna)
        else:
            return img, self.records[index].id_code

    def __len__(self):
        return self.len

    def item(self, index):
        return self.records[index].id_code


def test():
    DATA_DIR = 'data'

    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

    print(len(df))

    train_tsfm = T.Compose([
        T.RandomRotation(degrees=(-90, 90)),
        T.RandomVerticalFlip(0.25),
        T.RandomHorizontalFlip(0.25),
        T.ToTensor(),
    ])

    train_ds = RCICDefaultDataset(df, DATA_DIR, train_tsfm)

    print(torch.sum(train_ds[0][0]))

if __name__ == "__main__":
    test()