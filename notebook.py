#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import gc

import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR
from torch.autograd import Variable

from torchvision import models, transforms as T

from pathlib import Path
import csv
from collections import defaultdict

import time
from typing import *
from tqdm import tqdm_notebook as tqdm
from fastprogress import master_bar, progress_bar

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


# ## Configs

# In[ ]:


TYPE = 0

MODEL_NAME = 'resnet34'

CELL_TYPE = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']

DATA_DIR = 'data'
PRETRAINED_MODEL_DIR = 'experiments/models'
VERSION = 'kernel2gcp'
device = 'cuda'
batch_size = 144

TIME_LIMIT = 9 * 60 * 60
global_start_time = time.time()


# AverageMeter to track results.

# In[ ]:


class AverageMeter:
    ''' Computes and stores the average and current value '''
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def has_time_run_out() -> bool:
    return time.time() - global_start_time > TIME_LIMIT - 1000  


# ## Create directory for saving

# In[ ]:


if not os.path.isdir('results'):
    os.mkdir('results')
        
logname = (f'results/log_{VERSION}_{TYPE}.csv')

SAVE_DIR = Path(f"bin")
SAVE_DIR.mkdir(exist_ok=True, parents=True)

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train_loss', 
                            'valid_loss', 'valid_accuracy', 
                            'combined_valid_accuracy'])


# ## Load data

# In[ ]:


train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))


# ## Split data

# In[ ]:


def split_experiments(df):
    df['cell_type'] = df['experiment'].apply(lambda o: o.split('-')[0])
    return df


def filter_experiments(df, cell_type):
    df = split_experiments(df)
    return df[df['cell_type'] == cell_type]


def train_val_exp_split(train, test):

    train['is_train'] = 1
    df = train.append(test).reset_index(drop=True)
    df['is_train'].fillna(0, inplace=True)

    df['cell_type'] = df['experiment'].apply(lambda o: o.split('-')[0])

    # record the count of test experiments based on cell type
    test_experiments = (df[(df['is_train'] == 0)]
                            .groupby('cell_type')['experiment']
                            .unique())
    test_experiments_cnt_dict = {k: len(v) for k, v in test_experiments.to_dict().items()}

    # record cell type and their corresponding experiments
    all_experiments = df.groupby('cell_type')['experiment'].unique()
    all_experiments_dict = all_experiments.to_dict()

    # for each cell type, get the experiments that
    # will be in the validation set, determined by half the len of 
    # the test experiments 
    val_exps = []
    for _, v in enumerate(all_experiments_dict):
        num_test = test_experiments_cnt_dict[v]
        num_valid = max(1, int(num_test/4)) + num_test
        
        val_exps += all_experiments_dict[v][-num_valid:-num_test],

    # flatten array
    val_exps = [e for exp in val_exps for e in exp]

    # train_df comprises of 
    # is_train = 1 and is not in val_exps
    train_df = df[(~df['experiment'].isin(val_exps)) & 
                  (df['is_train'] == 1)]
    val_df = df[df['experiment'].isin(val_exps)]

    return train_df, val_df


def train_valid_split(train, test=None, experiment_split=False, test_size=0.1):
    if not experiment_split:
        train_df, valid_df = train_test_split(train, test_size=test_size, 
                                              stratify=train.sirna, 
                                              random_state=42)
    else:
        train_df, valid_df = train_val_exp_split(train, test)

    return train_df, valid_df


# In[ ]:


train_df = filter_experiments(train_df, CELL_TYPE[TYPE])
test_df = filter_experiments(test_df, CELL_TYPE[TYPE])

train_df, valid_df = train_valid_split(train_df, test_df, True)


# In[ ]:


cell_dfs = (train_df, valid_df, test_df)
cell_splits = ('train', 'valid', 'test ')

for split, cell_df in zip(cell_splits, cell_dfs):
    exp_list = cell_df["experiment"].nunique()
    exp_len = len(cell_df)
    print(f'{split} len: {exp_len} \t unique experiments: {exp_list}')


# In[ ]:


# valid_df['sirna'].value_counts()


# ## Dataset

# In[ ]:


class ImagesDS(Dataset):
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
        
        if self.tsfm:
            img = self.tsfm(img)

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


# In[ ]:


def get_two_sites(df, tsfm, mode):
    ds_s1 = ImagesDS(df, 
                     DATA_DIR,
                     site=1,
                     tsfm=tsfm,
                     mode=mode)

    ds_s2 = ImagesDS(df, 
                     DATA_DIR,
                     site=2, 
                     tsfm=tsfm, 
                     mode=mode)

    ds = ConcatDataset([ds_s1, ds_s2])

    return ds


# In[ ]:


train_tsfm = T.Compose([
    T.RandomRotation(degrees=(-90, 90)),
    T.RandomVerticalFlip(),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])

test_tsfm = T.Compose([
    T.ToTensor(),
])

train_ds = get_two_sites(train_df, train_tsfm, 'train')
valid_ds = get_two_sites(valid_df, test_tsfm, 'train')
test_ds = get_two_sites(test_df, test_tsfm, 'test')


# In[ ]:


len(train_ds), len(valid_ds), len(test_ds),


# ## Dataloader

# In[ ]:


train_dl = DataLoader(train_ds, batch_size=batch_size, 
                      shuffle=True, num_workers=4, drop_last=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, 
                      shuffle=True, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=batch_size, 
                     shuffle=False, num_workers=4)


# ## Model

# In[ ]:


NUM_CLASSES = 1108

def pretrained_model(file_name, num_classes):
    m = models.resnet34(pretrained=False, num_classes=num_classes)    
    
    new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

    m.conv1 = new_conv
    
    m.load_state_dict(torch.load(os.path.join(PRETRAINED_MODEL_DIR, file_name)))
    
    return nn.Sequential(*list(m.children())[:-2])

# model = pretrained_model('rn34_best_model_17.pth')


# In[ ]:


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        return cosine


# In[ ]:


class BestFittingModel(nn.Module):
    def __init__(self, num_classes, extract_feature=False):
        super(BestFittingModel, self).__init__()
        self.model = pretrained_model('rn34_best_model_17.pth', num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.arc_margin_product = ArcMarginProduct(512, num_classes)
        self.EX = 1
        self.bn1 = nn.BatchNorm1d(1024 * self.EX)
        self.fc1 = nn.Linear(1024 * self.EX, 512 * self.EX)
        self.bn2 = nn.BatchNorm1d(512 * self.EX)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512 * self.EX, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.extract_feature = extract_feature

    def forward(self, x):
        e5 = self.model(x)
        x = torch.cat((nn.AdaptiveAvgPool2d(1)(e5), nn.AdaptiveMaxPool2d(1)(e5)), dim=1)
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        x = F.dropout(x, p=0.25)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.5)

        x = x.view(x.size(0), -1)

        x = self.fc2(x)
        feature = self.bn3(x)

        cosine = self.arc_margin_product(feature)
        if self.extract_feature:
            return cosine, feature
        else:
            return cosine


# In[ ]:


def eval_momentum(model):
    for name, child in model.named_children():
        if name.find('bn') != -1:
            child.track_running_stats = False
        elif name.find('layer') != -1:
            for block_name, block_child in child.named_children():
                for layer_name, layer in block_child.named_children():
                    if layer_name.find('bn') != -1:
                        layer.track_running_stats = False

                        
def train_momentum(model):
    for name, child in model.named_children():
        if name.find('bn') != -1:
            child.track_running_stats = True
        elif name.find('layer') != -1:
            for block_name, block_child in child.named_children():
                for layer_name, layer in block_child.named_children():
                    if layer_name.find('bn') != -1:
                        layer.track_running_stats = True


# In[ ]:


model = BestFittingModel(NUM_CLASSES)

if torch.cuda.is_available() and torch.cuda.device_count() > 1: 
    model = torch.nn.DataParallel(model)
model.to(device)

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f'{num_gpus} gpu(s) available!')


# metric_fc = ArcFaceLoss()
# metric_fc.to(device)

gc.collect()


# ## Loss function, optimizers, LR scheduler

# In[ ]:


class ArcFaceLoss(nn.modules.Module):
    def __init__(self,s=65.0,m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels, epoch=0):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cosine, labels)
        gamma=1
        loss=(loss1+gamma*loss2)/(1+gamma)
        return loss


# In[ ]:


class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range']                 and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


# In[ ]:


# loss function
criterion = ArcFaceLoss() # nn.CrossEntropyLoss() # AMSoftmaxLoss(512, NUM_CLASSES).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=3e-4)

# lr_scheduler
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)


# ## Mixup

# In[ ]:


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).cuda()
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def do_mixup(model, input_, target):
    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)

    inputs, targets_a, targets_b = map(Variable, (inputs,
                                                  targets_a, targets_b))
    output = model(inputs)

    loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
    
    return loss, output


# ## Averaging predictions

# In[ ]:


def weighted_preds(fc_dict):
    id_preds = {}
    
    for k, id_code in enumerate(fc_dict):
        weighted_preds =  fc_dict[id_code][0].detach().cpu()  +                           fc_dict[id_code][1].detach().cpu() 
        id_preds[id_code] = torch.argmax(weighted_preds).item()
    
    subm = pd.DataFrame(list(id_preds.items()),
                        columns=['id_code', 'predicted_sirna'])
    
    return subm # len(subm) = 19897


def combined_accuracy(valid_fc_dict, valid_df):
    valid_preds = weighted_preds(valid_fc_dict)

    valid_sirna = valid_df[['id_code', 'sirna']].copy()
    
    assert len(valid_preds) == len(valid_sirna)

    valid_compare_table = pd.merge(valid_preds, valid_sirna,
                                   left_on='id_code',
                                   right_on='id_code')

    combined_acc = accuracy_score(valid_compare_table['predicted_sirna'].values,
                                  valid_compare_table['sirna'].values)
    
    return combined_acc


# ## Train one epoch

# In[ ]:


def train(train_loader: Any, model: Any, criterion: Any, 
          optimizer: Any, mb: Any, lr_scheduler: Any,
          num_grad_acc: Any):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    train_momentum(model)

    num_steps = len(train_loader)

    end = time.time()

    for idx, (input_, id_codes, target) in enumerate(progress_bar(train_loader, parent=mb)):
        
        input_ = input_.to(device)
        target = target.to(device)
        
        output = model(input_)      
        loss = criterion(output, target)
        
        # loss = model(input, target)
        
        _, predicts = torch.max(output.detach(), dim=1)
        
        # predicts = predicts.squeeze()
        
        avg_score.update(accuracy_score(predicts.cpu().numpy(), target.cpu()))
        losses.update(loss.data.item(), input_.size(0))
        
        loss.backward()
        
        if num_grad_acc is None:
            optimizer.step()
            optimizer.zero_grad()
        elif (idx+1) % num_grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()
            
#         optimizer.zero_grad()        
#         optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if has_time_run_out():
            break        
            
    return losses.avg


# In[ ]:


def valid_inference(data_loader: Any, model: Any, mb: Any):
    valid_fc_dict = defaultdict(list)
    losses = AverageMeter()
    
    model.eval()
    eval_momentum(model)

    all_predicts, all_targets = [], []    
    
    with torch.no_grad():
        for i, data in enumerate(progress_bar(data_loader, parent=mb)):
            input_, id_codes, target = data            
            
            input_ = input_.to(device)
            target = target.to(device)

#             feature = model(input_)
#             output = metric_fc(feature, target)        
#             loss = criterion(output, target)

            output = model(input_)
            loss = criterion(output, target)        
            
            # loss = model(input_, target)

            losses.update(loss.data.item(), input_.size(0))            
            _, predicts = torch.max(output.detach(), dim=1)
            
            # predicts = predicts.squeeze()            
            
            for i in range(len(id_codes)):
                valid_fc_dict[id_codes[i]] += output[i],        
                    
            all_predicts.append(predicts)            
            all_targets.append(target)

    predicts = torch.cat(all_predicts)
    targets = torch.cat(all_targets)
    
    valid_accuracy = accuracy_score(predicts.cpu().numpy(), targets.cpu())
    
    subm = weighted_preds(valid_fc_dict)
    
    combined_valid_accuracy = combined_accuracy(valid_fc_dict, valid_df)

    return losses.avg, valid_accuracy, combined_valid_accuracy


# In[ ]:


NUM_EPOCHS = 50
best_model_accuracy, best_model_epoch, best_model = 0, 0, None

mb = master_bar(range(1, NUM_EPOCHS+1))
lr_records = []
valid_losses = []
train_losses = []

for epoch in mb:
    
    torch.cuda.empty_cache()
    
    train_loss = train(train_dl, model, 
                       criterion, optimizer, 
                       mb, lr_scheduler, num_grad_acc=None)
    
    train_logstr = (f'Epoch: {epoch}\t'
                    f'Train loss: {train_loss:.3f}\t')
    
    valid_loss, valid_accuracy, combined_valid_accuracy = valid_inference(valid_dl, model, mb) 
    
    valid_logstr = (f'Val loss: {valid_loss:.3f}\t'
                    f'Val accuracy: {valid_accuracy:.3f}\t'
                    f'Combined val accuracy: {combined_valid_accuracy:.3f}\t')
    
    lr_scheduler.step()
#     current_lr = lr_scheduler.get_lr()
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)    
#     lr_records.append(current_lr)
    
#     lr_logstr = f'lr: {current_lr}'

    print(train_logstr, valid_logstr)    
    
    # write to logfile
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, valid_loss, valid_accuracy, combined_valid_accuracy])
        
    if combined_valid_accuracy >= best_model_accuracy:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{VERSION}_best_model_{epoch}.pth"))
        best_model = model
        best_model_accuracy = combined_valid_accuracy        
        best_model_epoch = epoch
        
    if has_time_run_out():
        break


# In[ ]:


# plt.figure(figsize=(6,4))
# plt.plot(lr_records, train_losses, 'b--', label='train loss')
# plt.plot(lr_records, valid_losses, 'g--', label='valid loss')
# plt.legend()
# plt.show()


# In[ ]:


print('best model accuracy:', best_model_accuracy)


# ## Predict

# In[ ]:


test_fc_dict = defaultdict(list)

def test_inference(data_loader: Any, model: Any):

    model.eval()
    eval_momentum(model)

    all_targets = []
    
    with torch.no_grad():
        preds = np.empty(0)
        for i, data in enumerate(tqdm(data_loader)):
            
            input_, id_codes = data
            
            output = model(input_.cuda())            
            
            for i in range(len(output)):
                test_fc_dict[id_codes[i]] += output[i],
            
            idx = output.max(dim=-1)[1].cpu().numpy()
            preds = np.append(preds, idx, axis=0)            
            
    return preds


# In[ ]:


model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 
                                              f"{VERSION}_best_model_{best_model_epoch}.pth")))
test_preds = test_inference(test_dl, model)


# In[ ]:


submission = weighted_preds(test_fc_dict)

submission.head()


# In[ ]:


submission['predicted_sirna'].nunique()


# In[ ]:


submission.to_csv(f'{VERSION}_submission_{TYPE}.csv', index=False)

