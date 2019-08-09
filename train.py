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

from datasets import get_dataloader, get_dataframes
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
from tsfm import get_transform

from utils import * 
import utils.config
import utils.checkpoint
import utils.metrics


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


def train_one_epoch(config, logger, train_loader, model, criterion, optimizer, num_grad_acc, lr_scheduler):
    logger.info('training')

    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    train_momentum(model)

    num_steps = len(train_loader)

    end = time.time()

    for idx, data in enumerate(train_loader):
        input_, id_codes, target = data

        # if using gpu
        if torch.cuda.is_available():
            input_, target = input_.cuda(), target.cuda()
        
        output = model(input_)

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

        lr_str = ''
        if idx % config.train.log_freq == 0:
            logger.info(f'[{idx}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      # f'accuracy {avg_score.val:.4f} ({avg_score.avg:.4f})'
                        + lr_str)

    return losses.avg


def validate_one_epoch(config, logger, val_loader, model, criterion, valid_df):
    logger.info('validatation')
    
    losses = AverageMeter()
    
    model.eval()
    train_momentum(model, False)

    valid_fc_dict = defaultdict(list)

    num_steps = len(val_loader)

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            input_, id_codes, target = data

            # if using gpu
            if torch.cuda.is_available():
                input_, target = input_.cuda(), target.cuda()
                        
            output = model(input_)
            loss = criterion(output, target)
                        
            losses.update(loss.data.item(), input_.size(0))
            
            for i in range(len(id_codes)):
                valid_fc_dict[id_codes[i]] += output[i],
                
        lr_str = ''
        if idx % config.val.log_freq == 0:
            logger.info(f'[{idx}/{num_steps}]\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      # f'accuracy {avg_score.val:.4f} ({avg_score.avg:.4f})'
                        + lr_str)
    
    combined_valid_accuracy = utils.metrics.combined_accuracy(valid_fc_dict, valid_df)

    return losses.avg, combined_valid_accuracy
    

def train(config, model, valid_df, train_loader, val_loader, criterion, optimizer, lr_scheduler, logger, last_epoch):

    best_score = 0.0
    best_epoch = 0

    for epoch in tqdm(range(last_epoch + 1, config.train.num_epochs + 1)):
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        train_loss = train_one_epoch(config, logger, train_loader, 
                                     model, criterion, optimizer, 
                                     config.train.num_grad_acc, lr_scheduler)
    
        train_logstr = (f'Epoch: {epoch}\t'
                        f'Train loss: {train_loss:.3f}\t')
    
        val_loss, val_accuracy = validate_one_epoch(config, logger, val_loader, 
                                                    model, criterion, valid_df)
    
        valid_logstr = (f'Val loss: {val_loss:.3f}\t'
                        f'Val accuracy: {val_accuracy:.3f}')
    
        # SGDR
        if config.optimizer.name == 'cosine':
            lr_scheduler = get_scheduler(config, optimizer)

        # One cyclic lr
        elif config.optimizer.name == 'cyclic_lr':
            current_lr = lr_scheduler.get_lr()
            logger.info(current_lr[-1])                

        logger.info(train_logstr + valid_logstr)
    
        # save best score, model
        if val_accuracy > best_score:
            best_score = val_accuracy
            best_epoch = epoch

            filename = f'{config.setup.version}_e{epoch:02d}_{best_score:.04f}.pth'
            model_dir = config.saved.model_dir

            save_checkpoint(model_dir, filename, model, epoch, best_score, 
                            optimizer, save_arch=True, params=config)

            logger.info(f'A snapshot was saved to {filename}')

    logger.info(f'best score: {best_score:.3f}')


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

    logger.info(f'train_dl len: {len(train_loader)}')
    logger.info(f'valid_dl len: {len(val_loader)}')
    
    # model
    model = create_model(config)

    print(model)

    # optimizer
    optimizer = get_optimizer(config, model.parameters())

    # lr_scheduler
    lr_scheduler = get_scheduler(config, optimizer)

    # criterion    
    criterion = get_loss(config)

    print(criterion)
    
    last_epoch = 0
    
    if config.setup.stage == -1:
        only_train(config, model, valid_df, train_loader, val_loader, criterion, optimizer, lr_scheduler, logger, last_epoch)
    else:
        train(config, model, valid_df, train_loader, val_loader, criterion, optimizer, lr_scheduler, logger, last_epoch)
    

def only_train(config, model, valid_df, train_loader, val_loader, criterion, optimizer, lr_scheduler, logger, last_epoch):

    best_score = np.nan
    best_epoch = 0

    for epoch in tqdm(range(last_epoch + 1, config.train.num_epochs + 1)):
        
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        train_loss = train_one_epoch(config, logger, train_loader, 
                                     model, criterion, optimizer, 
                                     config.train.num_grad_acc, lr_scheduler)
    
        train_logstr = (f'Epoch: {epoch}\t'
                        f'Train loss: {train_loss:.3f}\t')
    
        # val_loss, val_accuracy = validate_one_epoch(config, logger, val_loader, 
        #                                             model, criterion, valid_df)
    
        # valid_logstr = (f'Val loss: {val_loss:.3f}\t'
        #                 f'Val accuracy: {val_accuracy:.3f}')
    
        # SGDR
        if config.optimizer.name == 'cosine':
            lr_scheduler = get_scheduler(config, optimizer)

        # One cyclic lr
        elif config.optimizer.name == 'cyclic_lr':
            current_lr = lr_scheduler.get_lr()
            logger.info(current_lr[-1])                

        logger.info(train_logstr)
    
        # save best score, model
        if train_loss < best_score:
            best_score = train_loss
            best_epoch = epoch

            filename = f'{config.setup.version}_e{epoch:02d}_{best_score:.04f}.pth'
            model_dir = config.saved.model_dir

            save_checkpoint(model_dir, filename, model, epoch, best_score, 
                            optimizer, save_arch=True, params=config)

            logger.info(f'A snapshot was saved to {filename}')

    logger.info(f'best score: {best_score:.3f}')


## END ##

def test_model(config):
    m = create_model(config)
    criterion = get_loss(config)

    print(m)
    print(criterion)

    input_ = torch.randn((16, 6, 224, 224))
    label_ = torch.tensor([1, 2, 3, 4] * 4)

    output = m(input_)    

    print(output.size())

    loss = criterion(output, label_)

    print(loss)
    
        
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

    print('complete!')


if __name__ == "__main__":
    main()