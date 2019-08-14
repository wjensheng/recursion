from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
from collections import defaultdict
from tqdm import tqdm
from fastprogress import master_bar, progress_bar
import time
from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import get_dataframes, get_datasets, get_dataloaders
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler, LRFinder
from tsfm import get_transform

from utils import * 
import utils.config
import utils.checkpoint
import utils.metrics

import wandb

def create_model(config):
    model = get_model(config)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1: 
        model = torch.nn.DataParallel(model)

    # use gpu
    if torch.cuda.is_available(): 
        model = model.cuda()

    return model


def train_momentum(model, train=True):
    if torch.cuda.device_count() > 1:
        model = model.module.backbone
    else:
        model = model.backbone
        for name, child in model.named_children():
            if isinstance(child, nn.BatchNorm2d):
                child.track_running_stats = train
            elif isinstance(child, nn.Sequential):
                for block_name, block_child in child.named_children():
                    for layer_name, layer in block_child.named_children():
                        if isinstance(layer, nn.BatchNorm2d):
                            layer.track_running_stats = train                


def train_one_epoch(config, train_loader, model, criterion, optimizer, lr_scheduler, mb):
    # logger.info('training')

    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    train_momentum(model)

    num_steps = len(train_loader)

    end = time.time()

    for idx, data in enumerate(progress_bar(train_loader, parent=mb)):
        input_, id_codes, target = data

        # if using gpu
        if torch.cuda.is_available():
            input_, target = input_.cuda(), target.cuda()
        
        output = model(input_)

        loss = criterion(output, target)
                
        _, predicts = torch.max(output.detach(), dim=1)
        
        losses.update(loss.data.item(), input_.size(0))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if config.scheduler.name == 'cosine':
            lr_scheduler.step()                

        batch_time.update(time.time() - end)
        end = time.time()
        
    if config.scheduler.name != 'cosine':
        lr_scheduler.step()

    return losses.avg


def validate_one_epoch(config, val_loader, model, criterion, valid_df, mb):
    # logger.info('validatation')
    
    losses = AverageMeter()
    
    model.eval()
    train_momentum(model, False)

    valid_fc_dict = defaultdict(list)

    num_steps = len(val_loader)

    with torch.no_grad():
        for idx, data in enumerate(progress_bar(val_loader, parent=mb)):
            input_, id_codes, target = data

            # if using gpu
            if torch.cuda.is_available():
                input_, target = input_.cuda(), target.cuda()
                        
            output = model(input_)
            loss = criterion(output, target)
                        
            losses.update(loss.data.item(), input_.size(0))
            
            for i in range(len(id_codes)):
                valid_fc_dict[id_codes[i]] += output[i],
                
    combined_valid_accuracy = utils.metrics.combined_accuracy(valid_fc_dict, valid_df)

    return losses.avg, combined_valid_accuracy
    

def test_inference(config, data_loader: Any, model: Any):

    model.eval()
    train_momentum(model, False)

    test_fc_dict = defaultdict(list)
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):

            input_, id_codes = data

            # if using gpu
            if torch.cuda.is_available():
                input_ = input_.cuda()
        
            output = model(input_) # TODO: fix for no loss
            
            for i in range(len(output)):
                test_fc_dict[id_codes[i]] += output[i],
            
    submission, all_classes_preds  = utils.metrics.weighted_preds(test_fc_dict)

    cell_type = config.setup.cell_type
    submission_pattern = config.submission.pattern

    submission_fn = f'{submission_pattern}_t{cell_type}.csv'
    submission.to_csv(os.path.join(config.submission.submission_dir, submission_fn), index=False)

    softmax_preds = all_classes_preds['predicted_sirna'].values

    softmax_preds_fn = f'class_{submission_pattern}_t{cell_type}.pt'
    torch.save(softmax_preds, os.path.join(config.submission.submission_dir, softmax_preds_fn))

    print('csv and pt files saved to {}!'.format(config.submission.submission_dir))


def train(config, model, valid_df, train_loader, val_loader, criterion, optimizer, lr_scheduler, last_epoch):

    best_score = 0.0
    best_epoch = 0
    best_model = None

    mb = master_bar(range(last_epoch + 1, config.train.num_epochs + 1))

    for epoch in mb:
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        train_loss = train_one_epoch(config, train_loader, model, criterion, optimizer, lr_scheduler, mb)
    
        val_loss, val_accuracy = validate_one_epoch(config, val_loader, model, criterion, valid_df, mb)
    
        # SGDR
        if config.scheduler.name == 'cosine':
            lr_scheduler = get_scheduler(config, optimizer)
        
        wandb.log({
            'Train loss': train_loss,
            'Valid loss': val_loss,
            'Valid accuracy': val_accuracy
        })
    
        # save best score, model
        if val_accuracy > best_score:
            best_score = val_accuracy
            best_epoch = epoch
            best_model = model

            
            # model_dir = config.saved.model_dir

            # save_checkpoint(model_dir, filename, model, epoch, best_score, 
            #                 optimizer, save_arch=True, params=config)

    # checkpoint = f't{config.setup.cell_type}_e{best_epoch:02d}_{best_score:.04f}.pth'

    # torch.save(best_model.state_dict(), os.path.join(saved.model_dir, checkpoint))

    return best_model


def run(config):

    wandb.init(project='recursion')
    wandb.config.update(config)

    pprint.PrettyPrinter(indent=2).pprint(config)

    # check gpu status
    check_cuda()

    # valid_df for combined_accuracy
    _, valid_df, _ = get_dataframes(config)

    # get dataloders
    train_loader, val_loader, test_loader = get_dataloaders(config)

    print(f'train_dl len: {len(train_loader)}')
    print(f'valid_dl len: {len(val_loader)}')
    
    # model
    model = create_model(config)
    wandb.watch(model)

    # optimizer
    optimizer = get_optimizer(config, model.parameters())
    print(optimizer)
    
    # criterion    
    criterion = get_loss(config)
    print(criterion)

    # lr_scheduler
    lr_scheduler = get_scheduler(config, optimizer)
    print(lr_scheduler)
    
    last_epoch = 0        
    best_model = train(config, model, valid_df, train_loader, val_loader, criterion, optimizer, lr_scheduler, last_epoch)

    # generate and save predictions
    if config.setup.run_test:
        test_inference(test_loader, best_model)
    
## END ##

def test_model(config):
    m = create_model(config)
    criterion = get_loss(config)

    print(m)
    print(criterion)

    for k, v in m.named_parameters():
        print(k)

    input_ = torch.randn((16, 6, 224, 224))
    label_ = torch.tensor([1, 2, 3, 4] * 4)

    output = m(input_)    

    print('output size:', output.size())

    loss = criterion(output, label_)

    print(loss)


def test_ds(config):
    tr, val, te = get_dataframes(config)

    print(len(tr), len(val), len(te))

    print(tr['sirna'].unique())
    print(val['sirna'].nunique())

    # train_dl, valid_dl, test_dl = get_dataloaders(config)
    # x = train_dl.dataset[0][0]
    # print(x)
    # print(torch.max(x))
    # print(torch.min(x))
    # print(torch.mean(x))

        
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

    if not os.path.exists(config.experiment_dir):
        os.makedirs(config.experiment_dir)    

    if not os.path.exists(config.saved.model_dir):
        os.makedirs(config.saved.model_dir)

    if not os.path.exists(config.submission.submission_dir):
        os.makedirs(config.submission.submission_dir)

    seed_everything()  

    run(config)
    # test_model(config)    
    # test_ds(config)
    print('complete!')


if __name__ == "__main__":
    main()