from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
from collections import defaultdict
from tqdm import tqdm

import time
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.optim import Optimizer

from datasets import get_dataloader, get_dataframes
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
from tsfm import get_transform
from models.loss import ContrastiveLoss

from utils import * # create_logger, AverageMeter, seed_everything, check_cuda, save_checkpoint
import utils.config
import utils.checkpoint
import utils.metrics # TODO: for combined accuracy 

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

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
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

def pretrained_model(config, num_classes):
    m = models.resnet34(pretrained=False, num_classes=num_classes)    
    
    new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

    m.conv1 = new_conv
    
    m.load_state_dict(torch.load(os.path.join(config.saved.model_dir, 
                                              config.saved.model)))
    
    return nn.Sequential(*list(m.children())[:-2])


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
        return cosine


class BestFittingModel(nn.Module):
    def __init__(self, config, num_classes, extract_feature=False):
        super(BestFittingModel, self).__init__()
        self.model = pretrained_model(config, num_classes)
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


def create_model(config):
    model = BestFittingModel(config, NUM_CLASSES)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1: 
        model = torch.nn.DataParallel(model)

    # use gpu
    if torch.cuda.is_available(): 
        model = model.cuda()

    return model

def train_momentum(model, train=True):
    for name, child in model.named_children():
        if name.find('bn') != -1:
            child.track_running_stats = train
        elif name.find('layer') != -1:
            for block_name, block_child in child.named_children():
                for layer_name, layer in block_child.named_children():
                    if layer_name.find('bn') != -1:
                        layer.track_running_stats = train


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


def train_one_epoch(config, logger, train_loader, model, criterion, optimizer, num_grad_acc, lr_scheduler):
    logger.info('training')

    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    train_momentum(model.module.backbone)

    num_steps = len(train_loader)

    end = time.time()

    for idx, data in enumerate(train_loader):
        input_, id_codes, target = data

        # if using gpu
        if torch.cuda.is_available():
            input_, target = input_.cuda(), target.cuda()
        
        output = model(input_, target)

        loss = criterion(output, target)
                
        _, predicts = torch.max(output.detach(), dim=1)
        
        losses.update(loss.data.item(), input_.size(0))

        loss.backward()
        
        if num_grad_acc == None:
            optimizer.step()
            optimizer.zero_grad()
        elif (idx+1) % num_grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()
                    
        lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        
        # if idx % config.train.log_freq == 0:
        #     logger.info(f'[{idx}/{num_steps}]\t'
        #                 f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #                 f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
        #               # f'accuracy {avg_score.val:.4f} ({avg_score.avg:.4f})'
        #                 + lr_str)

    return losses.avg


def validate_one_epoch(config, logger, val_loader, model, criterion, valid_df):
    logger.info('validatation')
    
    losses = AverageMeter()
    
    model.eval()
    train_momentum(model.module.backbone, False)

    valid_fc_dict = defaultdict(list)

    num_steps = len(val_loader)

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            input_, id_codes, target = data            

            # if using gpu
            if torch.cuda.is_available():
                input_, target = input_.cuda(), target.cuda()
                        
            output = model(input_, target)
            loss = criterion(output, target)
                        
            losses.update(loss.data.item(), input_.size(0))            
            _, predicts = torch.max(output.detach(), dim=1)
            
            for i in range(len(id_codes)):
                valid_fc_dict[id_codes[i]] += output[i],
                
        # if idx % config.valid.log_freq == 0:
        #     logger.info(f'[{idx}/{num_steps}]\t'
        #                 f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
        #               # f'accuracy {avg_score.val:.4f} ({avg_score.avg:.4f})'
        #                 + lr_str)
    
    combined_valid_accuracy = utils.metrics.combined_accuracy(valid_fc_dict, valid_df)

    return losses.avg, combined_valid_accuracy
    

def run(config):

    # create logger
    log_filename = f'log_training_{config.setup.version}.txt'
    logger = create_logger(os.path.join(config.experiment_dir, log_filename))

    logger.info('=' * 50)

    # check gpu status
    check_cuda(logger)

    # valid_df for combined_accuracy
    _, valid_df, _ = get_dataframes(config)

    # get dataloders
    train_loader, val_loader, test_loader = get_dataloader(config)

    # valid_dl len: {len(val_loader)}
    logger.info(f'train_dl len: {len(train_loader)}')
    logger.info(f'valid_dl len: {len(val_loader)}')
    
    # model
    model = create_model(config)

    print(model)

    # criterion
    criterion = ArcFaceLoss() # nn.CrossEntropyLoss() # AMSoftmaxLoss(512, NUM_CLASSES).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=1e-3)

    # lr_scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, 
                                     T_max=len(train_loader), 
                                     eta_min=3e-5)

    # lr_scheduler = CyclicLR(optimizer, 
    #                     base_lr=3e-5, 
    #                     max_lr=5e-3,
    #                     step_size=30)                                     
    
    last_epoch = 0
    best_score = 0.010
    best_epoch = 0

    for epoch in tqdm(range(last_epoch + 1, config.train.num_epochs + 1)):
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        train_loss = train_one_epoch(config, logger, train_loader, model, criterion, optimizer, config.train.num_grad_acc, lr_scheduler)
    
        train_logstr = (f'Epoch: {epoch}\t'
                        f'Train loss: {train_loss:.3f}\t')
    
        valid_loss, valid_accuracy = validate_one_epoch(config, logger, val_loader, model, criterion, valid_df)
    
        valid_logstr = (f'Val loss: {valid_loss:.3f}\t'
                        f'Val accuracy: {valid_accuracy:.3f}')
    
        # # SGDR
        # if config.optimizer.name == 'cosine':
        #     lr_scheduler = get_scheduler(config, optimizer)
        # # One cyclic lr
        # elif config.optimizer.name == 'cyclic_lr':
        #     current_lr = lr_scheduler.get_lr()
        #     logger.info(current_lr[-1])                

        # SGDR
        
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=len(train_dl), eta_min=3e-5)        

        # # Cyclic
        # lr_scheduler.batch)step()
        # logger.info(lr_scheduler.get_lr()[-1])

        logger.info(train_logstr + valid_logstr)
    
        if combined_valid_accuracy >= best_model_accuracy:
            # torch.save(model.state_dict(), os.path.join(config.saved.model_dir, f"{config.setup.version}_best_model_{epoch}.pth"))
            best_model = model
            best_model_accuracy = combined_valid_accuracy        
            best_model_epoch = epoch
            # logger.info(f'A snapshot was saved to {filename}')

    logger.info(f'best score: {best_score:.3f}')

    submission, all_classes_preds = test_inference(test_loader, best_model)

    print(submission['predicted_sirna'].nunique())

    fn = f'{config.setup.version}_submission_{config.setup.cell_type}.csv'

    submission.to_csv(os.path.join(submission.submission_dir, fn), index=False)

    all_classes_preds['predicted_sirna'] = all_classes_preds['predicted_sirna'].apply(lambda o: o.numpy())

    fn = 'classes_' + fn

    all_classes_preds.to_csv(os.path.join(submission.submission_dir, fn), index=False)



def test_inference(data_loader: Any, model: Any):

    test_fc_dict = defaultdict(list)

    model.eval()

    all_targets = []
    
    with torch.no_grad():
        preds = np.empty(0)
        for i, data in enumerate(tqdm(data_loader)):
            
            input_, id_codes = data            
            
            input_ = input_.to(device)

            output = model(input_)
                        
            _, predicts = torch.max(output.detach(), dim=1)
            
            for i in range(len(output)):
                test_fc_dict[id_codes[i]] += output[i],
            
    subm, all_classes_preds  = utils.metrics.weighted_preds(test_fc_dict)        
    
    return subm, all_classes_preds    

def test_model(config):
    m = create_model(config)
    print(m)
    input_ = torch.randn((16, 6, 224, 224))
    label_ = torch.randn((16, 6))
    print(m(input_, label_))

def test_loss(config):
    criterion = get_loss(config)
    # contrastive_loss = ContrastiveLoss(margin=0.7)
    input_ = torch.randn(64, 35, requires_grad=True)
    label_ = torch.Tensor([-1, 2, 0, 0, 0, 0, 0] * 5)
    output = criterion(input_, label_)
    print(output)
    
        
def parse_args():
    parser = argparse.ArgumentParser(description='RXRX')
    parser.add_argument('--config', 
                        help='model configuration file (YAML)', 
                        type=str, required=True)                        
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = utils.config.load_config(args.config, args)

    pprint.PrettyPrinter(indent=2).pprint(config)

    if not os.path.exists(config.experiment_dir):
        os.makedirs(config.experiment_dir)    

    if not os.path.exists(config.saved.model_dir):
        os.makedirs(config.saved.model_dir)

    if not os.path.exists(config.submission.submission_dir):
        os.makedirs(config.submission.submission_dir)    


    seed_everything()  

    run(config)
    # test_model(config)    
    # test_loss(config)

    print('complete!')


if __name__ == "__main__":
    main()
