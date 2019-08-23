from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd 

from collections import defaultdict
from easydict import EasyDict as edict

from sklearn.preprocessing import LabelEncoder

import torch.utils.data
import torch.utils.data.sampler
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models, transforms as T

from albumentations import Compose, RandomRotate90, Flip, Transpose, Resize, Normalize
from albumentations import RandomContrast, RandomBrightness, RandomGamma
from albumentations import Blur, MotionBlur, InvertImg
from albumentations import Rotate, ShiftScaleRotate, RandomScale
from albumentations import GridDistortion, ElasticTransform

from .default import DefaultDataset

CELL_TYPE = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']

def get_two_sites(config, df, tsfm, mode):
    ds_s1 = DefaultDataset(df, 
                           config.data.data_dir,
                           site=1,
                           transform=tsfm,
                           mode=mode)

    ds_s2 = DefaultDataset(df, 
                           config.data.data_dir,
                           site=2,
                           transform=tsfm,
                           mode=mode)

    ds = ConcatDataset([ds_s1, ds_s2])

    return ds

def create_train_test(train_csv, test_csv, plate):
    plate_groups = np.zeros((1108,4), int)
    for sirna in range(1108):
        grp = train_csv.loc[train_csv.sirna==sirna,:].plate.value_counts().index.values
        assert len(grp) == 3
        plate_groups[sirna,0:3] = grp
        plate_groups[sirna,3] = 10 - grp.sum() # 1 + 2 + 3 + 4 = 10        

    d = defaultdict(list)
    sequence = plate_groups[:,3]

    for i, x in enumerate(sequence):
        d[x].append(i)
    
    test_df = test_csv[test_csv['plate'] == plate]
    train_df = train_csv[train_csv['sirna'].isin(d[plate])]

    le = LabelEncoder()

    train_df['sirna'] = le.fit_transform(train_df['sirna'])

    return train_df, test_df
    

def manual_split(df):    
    last_batch = ['HEPG2-07', 'HUVEC-15', 'HUVEC-16', 'RPE-07', 'U2OS-03']
    valid_df = df[df['experiment'].isin(last_batch)]
    train_df = df[~df['experiment'].isin(last_batch)]
    return train_df, valid_df  


def filter_experiments(df, cell_type):
    df['cell_type'] = df['experiment'].apply(lambda o: o.split('-')[0])
    return df[df['cell_type'] == cell_type]


def get_dataframes(config):
    train_df = pd.read_csv(os.path.join(config.data.data_dir, 
                                        config.data.train))
    test_df = pd.read_csv(os.path.join(config.data.data_dir, 
                                       config.  data.test))

    # stage 1: train on all dataset, valid on last batches
    if config.setup.stage:
        train_df = filter_experiments(train_df, CELL_TYPE[config.setup.cell_type])
        test_df = filter_experiments(test_df, CELL_TYPE[config.setup.cell_type])
    
    train_df, valid_df = manual_split(train_df)

    print('train_df----valid_df----test_df')
    print(len(train_df), '\t----', len(valid_df), '  ----', len(test_df))

    return train_df, valid_df, test_df


def get_datasets(config):
    SIZE = config.model.image_size

    train_transform = Compose([
        RandomRotate90(),
        Flip(),
        Resize(height=SIZE, width=SIZE, always_apply=True)
    ])  

    test_transform = Compose([
        Resize(height=SIZE, width=SIZE, always_apply=True)
    ])

    train_df, valid_df, test_df = get_dataframes(config)

    train_ds = get_two_sites(config, train_df, train_transform, 'train')
    valid_ds = get_two_sites(config, valid_df, test_transform, 'train')
    test_ds = get_two_sites(config, test_df, test_transform, 'test')

    return train_ds, valid_ds, test_ds


def get_dataloaders(config):
    train_ds, valid_ds, test_ds = get_datasets(config)

    train_dl = DataLoader(train_ds, shuffle=True,
                          batch_size=config.train.batch_size,
                          drop_last=True,
                          num_workers=config.num_workers,
                          pin_memory=False)

    valid_dl = DataLoader(valid_ds, shuffle=True,
                          batch_size=config.val.batch_size,
                          drop_last=False,
                          num_workers=config.num_workers,
                          pin_memory=False)

    test_dl = DataLoader(test_ds, shuffle=False,
                         batch_size=config.test.batch_size,
                         drop_last=False,
                         num_workers=config.num_workers,
                         pin_memory=False)                                                    
    
    return train_dl, valid_dl, test_dl



# def get_dataset(config, split, transform=None, last_epoch=-1):
#     f = globals().get(config.name)

#     return f(config.dir,
#              split=split,
#              transform=transform,
#              **config.params)


# def get_dataloader(config, split, transform=None, **_):
#     dataset = get_dataset(config.data, split, transform)

#     is_train = 'train' == split
#     batch_size = config.train.batch_size if is_train else config.eval.batch_size

#     dataloader = DataLoader(dataset,
#                             shuffle=is_train,
#                             batch_size=batch_size,
#                             drop_last=is_train,
#                             num_workers=config.transform.num_preprocessor,
#                             pin_memory=False)
#     return dataloader
