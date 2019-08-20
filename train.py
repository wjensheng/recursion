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
from schedulers import get_scheduler
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

    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    train_momentum(model)

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

        if config.train.num_grad_acc is None:
            optimizer.step()
            optimizer.zero_grad()
        elif (idx+1) % config.train.num_grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if config.scheduler.name == 'cosine':
            lr_scheduler.step()                
        
    if config.scheduler.name != 'cosine':
        lr_scheduler.step()

    return losses.avg


def validate_one_epoch(config, val_loader, model, criterion, valid_df, mb):
    
    losses = AverageMeter()
    
    model.eval()
    train_momentum(model, False)

    valid_fc_dict = defaultdict(list)

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
        
            output = model(input_)
            
            for i in range(len(output)):
                test_fc_dict[id_codes[i]] += output[i],
            
    submission, all_classes_preds  = utils.metrics.weighted_preds(test_fc_dict)

    return submission, all_classes_preds


def save_predictions(config, submission, all_classes_preds):

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
                    
    checkpoint = f't{config.setup.cell_type}_e{best_epoch:02d}_{best_score:.04f}.pth'
    torch.save(best_model.state_dict(), os.path.join(config.saved.model_dir, checkpoint))

    return best_model


def run(config):

    wandb.init(project=config.setup.project)
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
        submission, all_classes_preds = test_inference(config, test_loader, best_model)
        save_predictions(config, submission, all_classes_preds)
    
## END ##

def test_model(config):
    m = create_model(config)
    criterion = get_loss(config)

    print(m)
    print(criterion)

    # layers = list(criterion.named_parameters()) + \
    #          list(set(m.named_parameters()) - set(m.backbone.named_parameters()))
    # for l in layers:
    #     print(l)
    
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

        
def parse_args():
    parser = argparse.ArgumentParser(description='RXRX')
    parser.add_argument('--config', 
                        help='model configuration file (YAML)', 
                        type=str, required=True)
    # parser.add_argument("--image_size", 
    #                     dest='image_size', help="size of an image", 
    #                     type=int, default=320)
    # parser.add_argument("--num_epochs", 
    #                     dest='num_epochs', help="number of epochs to train", 
    #                     type=int, default=30)
    # parser.add_argument("--loss", 
    #                     dest='loss', help="loss function", 
    #                     type=str)
    # parser.add_argument("--optim_lr", 
    #                     dest='optim_lr', help="learning rate for optimizer", 
    #                     type=float)
    # parser.add_argument("--optim_wd", 
    #                     dest='optim_wd', help="weight decay for optimizer", 
    #                     type=float)
    # parser.add_argument("--eta_min", 
    #                     dest='eta_min', help="eta min for SGDR", 
    #                     type=float)
    # parser.add_argument("--t_max", 
    #                     dest='t_max', help="T max for SGDR", 
    #                     type=float)
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

    pprint.PrettyPrinter(indent=2).pprint(config)

    # run(config)
    test_model(config)    
    # test_ds(config)

    print('complete!')


if __name__ == "__main__":
    main()