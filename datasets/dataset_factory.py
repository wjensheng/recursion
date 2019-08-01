from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch.utils.data
import torch.utils.data.sampler
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models, transforms as T

from .default import RCICDefaultDataset
from .split import *

from easydict import EasyDict as edict

# TODO
# -convert tsfm to args
# -fix duplication

def get_two_sites(config, df, tsfm, mode):
    ds_s1 = RCICDefaultDataset(df, 
                               config.data.data_dir,
                               site=1,
                               tsfm=tsfm,
                               mode=mode)

    ds_s2 = RCICDefaultDataset(df, 
                               config.data.data_dir,
                               site=2, 
                               tsfm=tsfm, 
                               mode=mode)

    ds = ConcatDataset([ds_s1, ds_s2])

    return ds



def get_dataset(config):
    train_csv = 'train_small.csv' if config.setup.use_small else 'train.csv'
    train_filename = 'tran_small' if config.setup.use_small else 'train'
    
    train_df = pd.read_csv(os.path.join(config.data.data_dir, train_csv))
    test_df = pd.read_csv(os.path.join(config.data.data_dir, 'test.csv'))

    train_tsfm = T.Compose([
        # T.RandomRotation(degrees=(-90, 90)),
        # T.RandomVerticalFlip(),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    test_tsfm = T.Compose([
        T.ToTensor(),
    ])

    if config.setup.stage == 0:
        if not config.setup.combine:
            train_ds = RCICDefaultDataset(train_df, 
                                          config.data.data_dir,
                                          site=1, 
                                          tsfm=train_tsfm, 
                                          mode=train_filename)
        else: # config.setup.combine:
            train_ds = get_two_sites(config, train_df, train_tsfm, train_filename)


        valid_ds = test_ds = None
        
    elif config.setup.stage == 1:
        assert not config.setup.use_small, "Must use full dataset!"
        train_df, valid_df = manual_split(train_df)

        if not config.setup.combine:
            train_ds = RCICDefaultDataset(train_df, 
                                          config.data.data_dir, 
                                          tsfm=train_tsfm, 
                                          mode=train_filename)
            valid_ds = RCICDefaultDataset(valid_df, 
                                          config.data.data_dir, 
                                          tsfm=test_tsfm, 
                                          mode=train_filename)
        else:
            train_ds = get_two_sites(config, train_df, train_tsfm, train_filename)
            valid_ds = get_two_sites(config, valid_df, test_tsfm, train_filename)

        test_ds = None            

    elif config.setup.stage == 2:
        train_df, valid_df, test_df = train_valid_test(train=train_df, 
                                                       test=test_df,
                                                       split=config.setup.cell_type,
                                                       test_size=config.setup.test_size)

        if not config.setup.combine:                                                   
            train_ds = RCICDefaultDataset(train_df, 
                                          config.data.data_dir, 
                                          tsfm=train_tsfm, 
                                          mode=train_filename)
            valid_ds = RCICDefaultDataset(valid_df, 
                                          config.data.data_dir, 
                                          tsfm=test_tsfm, 
                                          mode=train_filename)
            test_ds = RCICDefaultDataset(test_df, 
                                         config.data.data_dir, 
                                         tsfm=test_tsfm, 
                                         mode='test')

        else: # config.setup.combine:
            train_ds = get_two_sites(config, train_df, train_tsfm, train_filename)
            valid_ds = get_two_sites(config, valid_df, test_tsfm, train_filename)
            test_ds = get_two_sites(config, test_df, test_tsfm, 'test')
                                  
    return train_ds, valid_ds, test_ds


def get_dataloader(config):    
    train_ds, valid_ds, test_ds = get_dataset(config)

    train_dl = DataLoader(train_ds, shuffle=True,
                          batch_size=config.train.batch_size,
                          drop_last=True,
                          num_workers=config.num_workers,
                          pin_memory=False)

    valid_dl = DataLoader(valid_ds, shuffle=False,
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

if __name__ == "__main__":
    config = edict()
    config.data = edict()
    config.data.data_dir = 'data'
    config.setup = edict()
    config.setup.use_small = False
    config.setup.stage = 2
    config.setup.cell_type = -1
    config.setup.combine = True

    train, valid, test = get_dataset(config)

    print(len(train), len(valid), len(test))