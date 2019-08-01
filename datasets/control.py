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

def controls_id_code(controls_df, experiment, plate, well_type):
    '''For a given experiment and plate,
    returns the id_code of the control cell.'''
    control_ids = controls_df[
        (controls_df['experiment'] == experiment) & 
        (controls_df['plate'] == int(plate)) &
        (controls_df['well_type'] == well_type)
    ]['id_code'].tolist()
    
    return control_ids

def control_pixels(controls_df, id_code):
    '''Returns negative and positive controls given a
    noncontrol id_code.'''
    experiment, plate, well = id_code.split('_')
    
    neg_ids = controls_id_code(controls_df, experiment, plate, 'negative_control')
    pos_ids = controls_id_code(controls_df, experiment, plate, 'positive_control')
    
    return neg_ids[0]

class RCICControlDataset(Dataset):
    def __init__(self, df, img_dir, controls_df, tsfm=None, mode='train', site=1, channels=[1,2,3,4,5,6]):        
        self.records = df.to_records(index=False)
        self.channels = channels
        self.site = site
        self.mode = mode
        self.img_dir = img_dir
        self.len = df.shape[0]
        self.tsfm = tsfm
        self.controls_df = controls_df
        
    def _load_img_as_tensor(self, file_name):
        img = Image.open(file_name)        
        
        if self.tsfm:
            img = self.tsfm(img)

        return img

    def _get_img_path(self, index, channel):
        experiment, plate, well = self.records[index].experiment, self.records[index].plate, self.records[index].well        
        noncontrol_path = '/'.join([self.img_dir,
                                    self.mode,
                                    experiment,
                                    f'Plate{plate}',
                                    f'{well}_s{self.site}_w{channel}.png'])
        
        return noncontrol_path
    
    def _get_control_img_path(self, index, channel):
        control_id_code = control_pixels(self.controls_df, 
                                         self.records[index].id_code)
        
        experiment, plate, well = control_id_code.split('_')
        control_path = '/'.join([self.img_dir,
                                 self.mode,
                                 experiment,
                                 f'Plate{plate}',
                                 f'{well}_s{self.site}_w{channel}.png'])
        
        return control_path
    
    def __getitem__(self, index):
        noncontrol_paths = [self._get_img_path(index, ch) for ch in self.channels]
        control_paths = [self._get_control_img_path(index, ch) for ch in self.channels]
        
        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in noncontrol_paths])
        control_img = torch.cat([self._load_img_as_tensor(img_path) for img_path in control_paths])
        
        if self.mode == 'train':
            return img, control_img, self.records[index].id_code, int(self.records[index].sirna)
        else:
            return img, control_img, self.records[index].id_code


    def __len__(self):
        return self.len


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

    train_ds = RCICControlDataset(df, DATA_DIR, train_tsfm)

    print(torch.sum(train_ds[0][0]))

if __name__ == "__main__":
    test()