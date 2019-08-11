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


###### KERNEL

# import os
# import gc

# import math
# import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt

# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Optimizer
# from torch.nn import Parameter
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
# from torch.autograd import Variable

# from torchvision import models, transforms as T

# from pathlib import Path
# import csv
# from collections import defaultdict

# import time
# from typing import *
# from tqdm import tqdm_notebook as tqdm
# from fastprogress import master_bar, progress_bar

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# import warnings
# warnings.filterwarnings('ignore')


# # ## Configs

# TYPE = 0

# SIZE = 224

# MODEL_NAME = 'resnet34'

# CELL_TYPE = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']

# DATA_DIR = 'data'
# PRETRAINED_MODEL_DIR = 'experiments/models'
# VERSION = 'kernel2gcp2'
# device = 'cuda'
# batch_size = 256

# TIME_LIMIT = 9 * 60 * 60
# global_start_time = time.time()


# # AverageMeter to track results.

# class AverageMeter:
#     ''' Computes and stores the average and current value '''
#     def __init__(self) -> None:
#         self.reset()

#     def reset(self) -> None:
#         self.val = 0.0
#         self.avg = 0.0
#         self.sum = 0.0
#         self.count = 0

#     def update(self, val: float, n: int = 1) -> None:
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
        
# def has_time_run_out() -> bool:
#     return time.time() - global_start_time > TIME_LIMIT - 1000  


# # ## Create directory for saving

# if not os.path.isdir('results'):
#     os.mkdir('results')
        
# logname = (f'results/log_{VERSION}_{TYPE}.csv')

# SAVE_DIR = Path(f"bin")
# SAVE_DIR.mkdir(exist_ok=True, parents=True)

# if not os.path.exists(logname):
#     with open(logname, 'w') as logfile:
#         logwriter = csv.writer(logfile, delimiter=',')
#         logwriter.writerow(['epoch', 'train_loss', 
#                             'valid_loss', 'valid_accuracy', 
#                             'combined_valid_accuracy'])


# # ## Load data


# train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
# test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))


# # ## Split data


# def split_experiments(df):
#     df['cell_type'] = df['experiment'].apply(lambda o: o.split('-')[0])
#     return df


# def filter_experiments(df, cell_type):
#     df = split_experiments(df)
#     return df[df['cell_type'] == cell_type]


# def train_val_exp_split(train, test):

#     train['is_train'] = 1
#     df = train.append(test).reset_index(drop=True)
#     df['is_train'].fillna(0, inplace=True)

#     df['cell_type'] = df['experiment'].apply(lambda o: o.split('-')[0])

#     # record the count of test experiments based on cell type
#     test_experiments = (df[(df['is_train'] == 0)]
#                             .groupby('cell_type')['experiment']
#                             .unique())
#     test_experiments_cnt_dict = {k: len(v) for k, v in test_experiments.to_dict().items()}

#     # record cell type and their corresponding experiments
#     all_experiments = df.groupby('cell_type')['experiment'].unique()
#     all_experiments_dict = all_experiments.to_dict()

#     # for each cell type, get the experiments that
#     # will be in the validation set, determined by half the len of 
#     # the test experiments 
#     val_exps = []
#     for _, v in enumerate(all_experiments_dict):
#         num_test = test_experiments_cnt_dict[v]
#         num_valid = max(1, int(num_test/4)) + num_test
        
#         val_exps += all_experiments_dict[v][-num_valid:-num_test],

#     # flatten array
#     val_exps = [e for exp in val_exps for e in exp]

#     # train_df comprises of 
#     # is_train = 1 and is not in val_exps
#     train_df = df[(~df['experiment'].isin(val_exps)) & 
#                   (df['is_train'] == 1)]
#     val_df = df[df['experiment'].isin(val_exps)]

#     return train_df, val_df


# def train_valid_split(train, test=None, experiment_split=False, test_size=0.1):
#     if not experiment_split:
#         train_df, valid_df = train_test_split(train, test_size=test_size, 
#                                               stratify=train.sirna, 
#                                               random_state=42)
#     else:
#         train_df, valid_df = train_val_exp_split(train, test)

#     return train_df, valid_df



# train_df = filter_experiments(train_df, CELL_TYPE[TYPE])
# test_df = filter_experiments(test_df, CELL_TYPE[TYPE])

# train_df, valid_df = train_valid_split(train_df, test_df, True)


# cell_dfs = (train_df, valid_df, test_df)
# cell_splits = ('train', 'valid', 'test ')

# for split, cell_df in zip(cell_splits, cell_dfs):
#     exp_list = cell_df["experiment"].nunique()
#     exp_len = len(cell_df)
#     print(f'{split} len: {exp_len} \t unique experiments: {exp_list}')



# # ## Dataset


# class ImagesDS(Dataset):
#     def __init__(self, df, img_dir, tsfm=None, mode='train', site=1, channels=[1,2,3,4,5,6]):        
#         self.records = df.to_records(index=False)
#         self.channels = channels
#         self.site = site
#         self.mode = mode
#         self.img_dir = img_dir
#         self.len = df.shape[0]
#         self.tsfm = tsfm
        
#     def _load_img_as_tensor(self, file_name):
#         img = Image.open(file_name)        
        
#         if self.tsfm:
#             img = self.tsfm(img)

#         return img

#     def _get_img_path(self, index, channel):
#         experiment, plate, well = self.records[index].experiment, self.records[index].plate, self.records[index].well
#         return '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site}_w{channel}.png'])
    
        
#     def __getitem__(self, index):
#         paths = [self._get_img_path(index, ch) for ch in self.channels]        
#         img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])
        
#         if self.mode == 'train':
#             return img, self.records[index].id_code, int(self.records[index].sirna)
#         else:
#             return img, self.records[index].id_code


#     def __len__(self):
#         return self.len


# def get_two_sites(df, tsfm, mode):
#     ds_s1 = ImagesDS(df, 
#                      DATA_DIR,
#                      site=1,
#                      tsfm=tsfm,
#                      mode=mode)

#     ds_s2 = ImagesDS(df, 
#                      DATA_DIR,
#                      site=2, 
#                      tsfm=tsfm, 
#                      mode=mode)

#     ds = ConcatDataset([ds_s1, ds_s2])

#     return ds



# train_tsfm = T.Compose([
#     T.RandomRotation(degrees=(-90, 90)),
#     T.RandomVerticalFlip(),
#     T.RandomHorizontalFlip(),
#     T.Resize((SIZE, SIZE)),
#     T.ToTensor(),
# ])

# test_tsfm = T.Compose([
#     T.Resize((SIZE, SIZE)),
#     T.ToTensor(),
# ])

# train_ds = get_two_sites(train_df, train_tsfm, 'train')
# valid_ds = get_two_sites(valid_df, test_tsfm, 'train')
# test_ds = get_two_sites(test_df, test_tsfm, 'test')


# len(train_ds), len(valid_ds), len(test_ds),

# train_dl = DataLoader(train_ds, batch_size=batch_size, 
#                       shuffle=True, num_workers=4, drop_last=True)
# valid_dl = DataLoader(valid_ds, batch_size=batch_size, 
#                       shuffle=True, num_workers=4)
# test_dl = DataLoader(test_ds, batch_size=batch_size, 
#                      shuffle=False, num_workers=4)


# # ## Model


# NUM_CLASSES = 1108

# def pretrained_model(file_name, num_classes):
#     m = models.resnet34(pretrained=False, num_classes=num_classes)    
    
#     new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

#     m.conv1 = new_conv
    
#     m.load_state_dict(torch.load(os.path.join(PRETRAINED_MODEL_DIR, file_name)))
    
#     return nn.Sequential(*list(m.children())[:-2])

# # model = pretrained_model('rn34_best_model_17.pth')


# class ArcMarginProduct(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(ArcMarginProduct, self).__init__()
#         self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)

#     def forward(self, features):
#         cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
#         return cosine



# class BestFittingModel(nn.Module):
#     def __init__(self, num_classes, extract_feature=False):
#         super(BestFittingModel, self).__init__()
#         self.model = pretrained_model('rn34_best_model_17.pth', num_classes)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.arc_margin_product = ArcMarginProduct(512, num_classes)
#         self.EX = 1
#         self.bn1 = nn.BatchNorm1d(1024 * self.EX)
#         self.fc1 = nn.Linear(1024 * self.EX, 512 * self.EX)
#         self.bn2 = nn.BatchNorm1d(512 * self.EX)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(512 * self.EX, 512)
#         self.bn3 = nn.BatchNorm1d(512)
#         self.extract_feature = extract_feature

#     def forward(self, x):
#         e5 = self.model(x)
#         x = torch.cat((nn.AdaptiveAvgPool2d(1)(e5), nn.AdaptiveMaxPool2d(1)(e5)), dim=1)
#         x = x.view(x.size(0), -1)
#         x = self.bn1(x)
#         x = F.dropout(x, p=0.25)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.bn2(x)
#         x = F.dropout(x, p=0.5)

#         x = x.view(x.size(0), -1)

#         x = self.fc2(x)
#         feature = self.bn3(x)

#         cosine = self.arc_margin_product(feature)
#         if self.extract_feature:
#             return cosine, feature
#         else:
#             return cosine


# def eval_momentum(model):
#     for name, child in model.named_children():
#         if name.find('bn') != -1:
#             child.track_running_stats = False
#         elif name.find('layer') != -1:
#             for block_name, block_child in child.named_children():
#                 for layer_name, layer in block_child.named_children():
#                     if layer_name.find('bn') != -1:
#                         layer.track_running_stats = False

                        
# def train_momentum(model):
#     for name, child in model.named_children():
#         if name.find('bn') != -1:
#             child.track_running_stats = True
#         elif name.find('layer') != -1:
#             for block_name, block_child in child.named_children():
#                 for layer_name, layer in block_child.named_children():
#                     if layer_name.find('bn') != -1:
#                         layer.track_running_stats = True



# model = BestFittingModel(NUM_CLASSES)

# if torch.cuda.is_available() and torch.cuda.device_count() > 1: 
#     model = torch.nn.DataParallel(model)
# model.to(device)

# print(model)

# gc.collect()


# # ## Loss function, optimizers, LR scheduler

# class ArcFaceLoss(nn.modules.Module):
#     def __init__(self,s=65.0,m=0.5):
#         super(ArcFaceLoss, self).__init__()
#         self.classify_loss = nn.CrossEntropyLoss()
#         self.s = s
#         self.easy_margin = False
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m

#     def forward(self, logits, labels, epoch=0):
#         cosine = logits
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         phi = cosine * self.cos_m - sine * self.sin_m
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)

#         one_hot = torch.zeros(cosine.size(), device='cuda')
#         one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         output *= self.s
#         loss1 = self.classify_loss(output, labels)
#         loss2 = self.classify_loss(cosine, labels)
#         gamma=1
#         loss=(loss1+gamma*loss2)/(1+gamma)
#         return loss



# class CyclicLR(object):
#     def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
#                  step_size=2000, mode='triangular', gamma=1.,
#                  scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

#         if not isinstance(optimizer, Optimizer):
#             raise TypeError('{} is not an Optimizer'.format(
#                 type(optimizer).__name__))
#         self.optimizer = optimizer

#         if isinstance(base_lr, list) or isinstance(base_lr, tuple):
#             if len(base_lr) != len(optimizer.param_groups):
#                 raise ValueError("expected {} base_lr, got {}".format(
#                     len(optimizer.param_groups), len(base_lr)))
#             self.base_lrs = list(base_lr)
#         else:
#             self.base_lrs = [base_lr] * len(optimizer.param_groups)

#         if isinstance(max_lr, list) or isinstance(max_lr, tuple):
#             if len(max_lr) != len(optimizer.param_groups):
#                 raise ValueError("expected {} max_lr, got {}".format(
#                     len(optimizer.param_groups), len(max_lr)))
#             self.max_lrs = list(max_lr)
#         else:
#             self.max_lrs = [max_lr] * len(optimizer.param_groups)

#         self.step_size = step_size

#         if mode not in ['triangular', 'triangular2', 'exp_range']                 and scale_fn is None:
#             raise ValueError('mode is invalid and scale_fn is None')

#         self.mode = mode
#         self.gamma = gamma

#         if scale_fn is None:
#             if self.mode == 'triangular':
#                 self.scale_fn = self._triangular_scale_fn
#                 self.scale_mode = 'cycle'
#             elif self.mode == 'triangular2':
#                 self.scale_fn = self._triangular2_scale_fn
#                 self.scale_mode = 'cycle'
#             elif self.mode == 'exp_range':
#                 self.scale_fn = self._exp_range_scale_fn
#                 self.scale_mode = 'iterations'
#         else:
#             self.scale_fn = scale_fn
#             self.scale_mode = scale_mode

#         self.batch_step(last_batch_iteration + 1)
#         self.last_batch_iteration = last_batch_iteration

#     def batch_step(self, batch_iteration=None):
#         if batch_iteration is None:
#             batch_iteration = self.last_batch_iteration + 1
#         self.last_batch_iteration = batch_iteration
#         for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
#             param_group['lr'] = lr

#     def _triangular_scale_fn(self, x):
#         return 1.

#     def _triangular2_scale_fn(self, x):
#         return 1 / (2. ** (x - 1))

#     def _exp_range_scale_fn(self, x):
#         return self.gamma**(x)

#     def get_lr(self):
#         step_size = float(self.step_size)
#         cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
#         x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

#         lrs = []
#         param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
#         for param_group, base_lr, max_lr in param_lrs:
#             base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
#             if self.scale_mode == 'cycle':
#                 lr = base_lr + base_height * self.scale_fn(cycle)
#             else:
#                 lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
#             lrs.append(lr)
#         return lrs



# # loss function
# criterion = ArcFaceLoss() # nn.CrossEntropyLoss() # AMSoftmaxLoss(512, NUM_CLASSES).to(device)

# # optimizer
# optimizer = torch.optim.Adam(model.parameters(), 
#                              lr=3e-3)

# # lr_scheduler
# # lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

# lr_scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dl), eta_min=3e-4)


# # ## Averaging predictions

# def weighted_preds(fc_dict):
#     id_preds = {}
#     classes_preds = {}
    
#     for k, id_code in enumerate(fc_dict):
#         weighted_preds =  fc_dict[id_code][0].detach().cpu()  +                           fc_dict[id_code][1].detach().cpu() 
#         id_preds[id_code] = torch.argmax(weighted_preds).item()
#         classes_preds[id_code] = weighted_preds
    
#     subm = pd.DataFrame(list(id_preds.items()),
#                         columns=['id_code', 'predicted_sirna'])
    
#     all_classes_preds = pd.DataFrame(list(classes_preds.items()),
#                                      columns=['id_code', 'predicted_sirna'])
    
#     return subm, all_classes_preds


# def combined_accuracy(valid_fc_dict, valid_df):
#     valid_preds, _ = weighted_preds(valid_fc_dict)

#     valid_sirna = valid_df[['id_code', 'sirna']].copy()
    
#     assert len(valid_preds) == len(valid_sirna)

#     valid_compare_table = pd.merge(valid_preds, valid_sirna,
#                                    left_on='id_code',
#                                    right_on='id_code')

#     combined_acc = accuracy_score(valid_compare_table['predicted_sirna'].values,
#                                   valid_compare_table['sirna'].values)
    
#     return combined_acc


# # ## Train one epoch

# def train(train_loader: Any, model: Any, criterion: Any, 
#           optimizer: Any, mb: Any, lr_scheduler: Any,
#           num_grad_acc: Any):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     avg_score = AverageMeter()

#     model.train()
#     train_momentum(model)

#     num_steps = len(train_loader)

#     end = time.time()

#     for idx, (input_, id_codes, target) in enumerate(progress_bar(train_loader, parent=mb)):
        
#         input_ = input_.to(device)
#         target = target.to(device)

#         output = model(input_)      
#         loss = criterion(output, target)
        
#         # loss = model(input, target)
        
#         _, predicts = torch.max(output.detach(), dim=1)
        
#         # predicts = predicts.squeeze()
        
#         avg_score.update(accuracy_score(predicts.cpu().numpy(), target.cpu()))
#         losses.update(loss.data.item(), input_.size(0))
        
#         loss.backward()
        
#         if num_grad_acc is None:
#             optimizer.step()
#             optimizer.zero_grad()
#         elif (idx+1) % num_grad_acc == 0:
#             optimizer.step()
#             optimizer.zero_grad()
            
# #         optimizer.zero_grad()        
# #         optimizer.step()

#         lr_scheduler.step()

#         batch_time.update(time.time() - end)
#         end = time.time()

#         if has_time_run_out():
#             break        
            
#     return losses.avg


# def valid_inference(data_loader: Any, model: Any, mb: Any):
#     valid_fc_dict = defaultdict(list)
#     losses = AverageMeter()
    
#     model.eval()
#     eval_momentum(model)

#     all_predicts, all_targets = [], []    
    
#     with torch.no_grad():
#         for i, data in enumerate(progress_bar(data_loader, parent=mb)):
#             input_, id_codes, target = data            
            
#             input_ = input_.to(device)
#             target = target.to(device)

# #             feature = model(input_)
# #             output = metric_fc(feature, target)        
# #             loss = criterion(output, target)

#             output = model(input_)
#             loss = criterion(output, target)        
            
#             # loss = model(input_, target)

#             losses.update(loss.data.item(), input_.size(0))            
#             _, predicts = torch.max(output.detach(), dim=1)
            
#             # predicts = predicts.squeeze()            
            
#             for i in range(len(id_codes)):
#                 valid_fc_dict[id_codes[i]] += output[i],        
                    
#             all_predicts.append(predicts)            
#             all_targets.append(target)

#     predicts = torch.cat(all_predicts)
#     targets = torch.cat(all_targets)
    
#     valid_accuracy = accuracy_score(predicts.cpu().numpy(), targets.cpu())
    
#     subm = weighted_preds(valid_fc_dict)
    
#     combined_valid_accuracy = combined_accuracy(valid_fc_dict, valid_df)

#     return losses.avg, valid_accuracy, combined_valid_accuracy


# NUM_EPOCHS = 30
# best_model_accuracy, best_model_epoch, best_model = 0, 0, None

# mb = master_bar(range(1, NUM_EPOCHS+1))
# lr_records = []
# valid_losses = []
# train_losses = []

# for epoch in mb:
    
#     torch.cuda.empty_cache()
    
#     train_loss = train(train_dl, model, 
#                        criterion, optimizer, 
#                        mb, lr_scheduler, num_grad_acc=None)
    
#     train_logstr = (f'Epoch: {epoch}\t'
#                     f'Train loss: {train_loss:.3f}\t')
    
#     valid_loss, valid_accuracy, combined_valid_accuracy = valid_inference(valid_dl, model, mb) 
    
#     valid_logstr = (f'Val loss: {valid_loss:.3f}\t'
#                     f'Val accuracy: {valid_accuracy:.3f}\t'
#                     f'Combined val accuracy: {combined_valid_accuracy:.3f}\t')

#     lr_scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dl), eta_min=3e-5)
    
#     # lr_scheduler.step()
# #     current_lr = lr_scheduler.get_lr()
    
#     # train_losses.append(train_loss)
#     # valid_losses.append(valid_loss)    
# #     lr_records.append(current_lr)
    
# #     lr_logstr = f'lr: {current_lr[-1]}'

#     print(train_logstr, valid_logstr)    
    
#     # write to logfile
#     with open(logname, 'a') as logfile:
#         logwriter = csv.writer(logfile, delimiter=',')
#         logwriter.writerow([epoch, train_loss, valid_loss, valid_accuracy, combined_valid_accuracy])
        
#     if combined_valid_accuracy >= best_model_accuracy:
#         torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{VERSION}_best_model_{epoch}.pth"))
#         best_model = model
#         best_model_accuracy = combined_valid_accuracy        
#         best_model_epoch = epoch
        
#     if has_time_run_out():
#         break


# # plt.figure(figsize=(6,4))
# # plt.plot(lr_records, train_losses, 'b--', label='train loss')
# # plt.plot(lr_records, valid_losses, 'g--', label='valid loss')
# # plt.legend()
# # plt.show()


# print('best model accuracy:', best_model_accuracy)


# # ## Predict

# test_fc_dict = defaultdict(list)

# def test_inference(data_loader: Any, model: Any):

#     model.eval()

#     all_targets = []
    
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(data_loader)):
            
#             noncontrols, id_codes, = data
            
#             # output = model(noncontrols.cuda(), controls.cuda())
#             output = model(noncontrols.cuda())
                        
#             _, predicts = torch.max(output.detach(), dim=1)
            
#             for i in range(len(output)):
#                 test_fc_dict[id_codes[i]] += output[i],
            
#     subm, all_classes_preds  = weighted_preds(test_fc_dict)        
    
#     return subm, all_classes_preds


# model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 
#                                               f"{VERSION}_best_model_{best_model_epoch}.pth")))
# submission, all_classes_preds = test_inference(test_dl, model)


# submission, all_classes_preds = weighted_preds(test_fc_dict)

# print(submission.head())


# print(submission['predicted_sirna'].nunique())


# submission.to_csv(f'{VERSION}_submission.csv', index=False)

# softmax_preds = all_classes_preds['predicted_sirna'].values
# torch.save(softmax_preds, f'class_{VERSION}_submission')


###### KERNEL



#####

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import numpy as np
# import torch.utils.data
# import torch.utils.data.sampler
# from torch.utils.data import DataLoader, ConcatDataset
# from torchvision import models, transforms as T

# from .default import DefaultDataset

# from easydict import EasyDict as edict

# def get_two_sites(config, df, mode):
#     if mode == 'train':
#         df = pd.read_csv(os.path.join(config.data.data_dir, 
#                                       config.data.train))

#         tsfm = Compose([
#             RandomRotate90(),
#             Resize(height=SIZE, width=SIZE, always_apply=True)
#         ])                                            

#     elif mode == 'valid':
#         df = pd.read_csv(os.path.join(config.data.data_dir, 
#                                         config.data.train))
#         df = manual_split(df)

#         tsfm = Compose([
#             Resize(height=SIZE, width=SIZE, always_apply=True)
#         ])

#     elif mode == 'test':
#         df = pd.read_csv(os.path.join(config.data.data_dir, 
#                                        config.data.test))

#         tsfm = Compose([
#             Resize(height=SIZE, width=SIZE, always_apply=True)
#         ])                                       

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


# def manual_split(df):
#     last_batch = ['HEPG2-07', 'HUVEC-16', 'RPE-07', 'U2OS-03']
#     valid_df = df[df['experiment'].isin(last_batch)]
#     # train_df = df[~df['experiment'].isin(last_batch)]
#     return valid_df  


# def get_dataframes(config):
#     train_df = pd.read_csv(os.path.join(config.data.data_dir, 
#                                         config.data.train))
#     test_df = pd.read_csv(os.path.join(config.data.data_dir, 
#                                        config.data.test))
#     train_df, valid_df = manual_split(train_df)
#     return train_df, valid_df, test_df


# # def get_datasets(config):
# #     SIZE = config.model.image_size

# #     train_df, valid_df, test_df = get_dataframes(config)

# #     train_ds = get_two_sites(config, train_df, train_transform, 'train')
# #     valid_ds = get_two_sites(config, valid_df, test_transform, 'train')
# #     test_ds = get_two_sites(config, test_df, test_transform, 'test')

# #     return train_ds, valid_ds, test_ds


# # def get_dataloaders(config):
# #     train_ds, valid_ds, test_ds = get_datasets(config)

# #     train_dl = DataLoader(train_ds, shuffle=True,
# #                           batch_size=config.train.batch_size,
# #                           drop_last=True,
# #                           num_workers=config.num_workers,
# #                           pin_memory=False)

# #     valid_dl = DataLoader(valid_ds, shuffle=True,
# #                           batch_size=config.val.batch_size,
# #                           drop_last=False,
# #                           num_workers=config.num_workers,
# #                           pin_memory=False)

# #     test_dl = DataLoader(test_ds, shuffle=False,
# #                          batch_size=config.test.batch_size,
# #                          drop_last=False,
# #                          num_workers=config.num_workers,
# #                          pin_memory=False)                                                    
    
# #     return train_dl, valid_dl, test_dl





# def get_dataset(config, split, transform=None):
#     return get_two_sites(config, transform, 'train')


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

# def test():
#     train_transform = Compose([
#         RandomRotate90(),
#         Resize(height=SIZE, width=SIZE, always_apply=True)
#     ]) 

#     test_transform = Compose([
#         Resize(height=SIZE, width=SIZE, always_apply=True)
#     ])

#     train_dl = get_dataloader(config, 'train', train_transform)

###


# def only_train(config, model, valid_df, train_loader, val_loader, criterion, optimizer, lr_scheduler, logger, last_epoch):

#     best_score = np.nan
#     best_epoch = 0

#     mb = master_bar(range(last_epoch + 1, config.train.num_epochs + 1))

#     for epoch in mb:
        
#         if torch.cuda.is_available(): torch.cuda.empty_cache()

#         train_loss = train_one_epoch(config, logger, train_loader, 
#                                      model, criterion, optimizer, 
#                                      config.train.num_grad_acc, lr_scheduler,
#                                      mb)
    
#         train_logstr = (f'Epoch: {epoch}\t'
#                         f'Train loss: {train_loss:.3f}\t')
    
#         # SGDR
#         if config.scheduler.name == 'cosine':
#             lr_scheduler = get_scheduler(config, optimizer)

#         # One cyclic lr
#         elif config.scheduler.name == 'cyclic':
#             current_lr = lr_scheduler.get_lr()
#             logger.info(current_lr[-1])                

#         logger.info(train_logstr)
    
#         # save best score, model
#         if train_loss < best_score:
#             best_score = train_loss
#             best_epoch = epoch

#             filename = f'{config.setup.version}_e{epoch:02d}_{best_score:.04f}.pth'
#             model_dir = config.saved.model_dir

#             save_checkpoint(model_dir, filename, model, epoch, best_score, 
#                             optimizer, save_arch=True, params=config)

#             logger.info(f'A snapshot was saved to {filename}')

#     logger.info(f'best score: {best_score:.3f}')

