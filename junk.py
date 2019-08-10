# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# from easydict import EasyDict as edict

# import numpy as np
# import torch.utils.data
# import torch.utils.data.sampler
# from torch.utils.data import DataLoader, ConcatDataset
# from torchvision import models, transforms as T

# from .default import RCICDefaultDataset
# from .data_utils import DefaultDataset
# from .split import *

# CELL_TYPE = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']

# def get_two_sites(config, df, tsfm, mode):
#     ds_s1 = DefaultDataset(df, 
#                            config.data.data_dir,
#                            site=1,
#                            tsfm=tsfm,
#                            mode=mode)

#     ds_s2 = DefaultDataset(df, 
#                            config.data.data_dir,
#                            site=2, 
#                            tsfm=tsfm, 
#                            mode=mode)

#     ds = ConcatDataset([ds_s1, ds_s2])

#     return ds


# def get_dataframes(config):    
#     train_df = pd.read_csv(os.path.join(config.data.data_dir, 
#                                         config.data.train))
#     test_df = pd.read_csv(os.path.join(config.data.data_dir, 
#                                        config.data.test))

#     # stage -1: no validation
#     if config.setup.stage == -1:
#         valid_df = test_df = None

#     # stage 0: train on all dataset, valid on last batches
#     elif config.setup.stage == 0:        
#         train_df, valid_df = manual_split(train_df)
#         test_df = None

#     # stage 1: validation set based on cell types
#     elif config.setup.stage == 1:
#         train_df = filter_experiments(train_df, CELL_TYPE[config.setup.cell_type])        
#         train_df, valid_df = manual_split(train_df)
#         test_df = filter_experiments(test_df, CELL_TYPE[config.setup.cell_type])

#     # stage 2: larger validation set
#     elif config.setup.stage == 2:
#         train_df, valid_df = train_val_exp_split(train=train_df, test=test_df)

#     else:
#         raise ValueError('Unknown stage!')    

#     return train_df, valid_df, test_df


# def get_datasets(config, train_tsfm, test_tsfm):
#     train_df, valid_df, test_df = get_dataframes(config)

#     # SIZE = config.model.image_size
#     # train_tsfm = T.Compose([        
#     #     T.RandomRotation(degrees=(-90, 90)),
#     #     T.RandomVerticalFlip(),
#     #     T.RandomHorizontalFlip(),
#     #     T.Resize((SIZE, SIZE)),
#     #     T.Normalize(mean=[0.485, 0.485, 0.456, 0.456, 0.406, 0.406],
#     #                 std=[0.229, 0.229, 0.225, 0.225, 0.224, 0.224]),
#     #     T.ToTensor(),
#     # ])

#     # test_tsfm = T.Compose([
#     #     T.Resize((SIZE, SIZE)),
#     #     T.Normalize(mean=[0.485, 0.485, 0.456, 0.456, 0.406, 0.406],
#     #                 std=[0.229, 0.229, 0.225, 0.225, 0.224, 0.224]),
#     #     T.ToTensor(),        
#     # ])

#     # stage -1: train on all dataset
#     if config.setup.stage == -1:
#         print('train experiments:', train_df['experiment'].unique())
#         train_ds = get_two_sites(config, train_df, train_tsfm, 'train')
#         valid_ds = test_ds = train_ds[0] # placeholder
    
#     # stage 0: valid on last experiments
#     elif config.setup.stage == 0:  
#         print('train experiments:', train_df['experiment'].unique())
#         print('valid experiments:', valid_df['experiment'].unique())        
#         train_ds = get_two_sites(config, train_df, train_tsfm, 'train')
#         valid_ds = get_two_sites(config, valid_df, test_tsfm, 'train')
#         test_ds = train_ds[0]

#     # stage 1 or 2: last batch validation set
#     else: # config.setup.stage == 1 or 2:
#         print('train experiments:', train_df['experiment'].unique())
#         print('valid experiments:', valid_df['experiment'].unique())
#         print('test experiments:', test_df['experiment'].unique())        
#         train_ds = get_two_sites(config, train_df, train_tsfm, 'train')
#         valid_ds = get_two_sites(config, valid_df, test_tsfm, 'train')
#         test_ds = get_two_sites(config, test_df, test_tsfm, 'train')
                            
#     return train_ds, valid_ds, test_ds


# def get_dataloader(config, train_tsfm, test_tsfm):    
#     train_ds, valid_ds, test_ds = get_datasets(config, train_tsfm, test_tsfm)

#     train_dl = DataLoader(train_ds, shuffle=True,
#                           batch_size=config.train.batch_size,
#                           drop_last=True,
#                           num_workers=config.num_workers,
#                           pin_memory=False)

#     valid_dl = DataLoader(valid_ds, shuffle=True,
#                           batch_size=config.val.batch_size,
#                           drop_last=False,
#                           num_workers=config.num_workers,
#                           pin_memory=False)

#     test_dl = DataLoader(test_ds, shuffle=False,
#                          batch_size=config.test.batch_size,
#                          drop_last=False,
#                          num_workers=config.num_workers,
#                          pin_memory=False)                                                    
    
#     return train_dl, valid_dl, test_dl


# if __name__ == "__main__":
#     config = edict()
#     config.data = edict()
#     config.data.data_dir = 'data'
#     config.setup = edict()
#     config.setup.use_small = False
#     config.setup.stage = 2
#     config.setup.cell_type = 3
#     config.setup.combine = True

#     train, valid, test = get_datasets(config)
#     train_dl, valid_dl, test_dl = get_dataset(config)

#     print(len(train), len(valid), len(test))
