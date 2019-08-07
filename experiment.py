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

from datasets import get_dataloader, get_dataframes
from models import get_model, pretrained_model, ArcMarginProduct, BestFittingModel, ArcFaceLoss
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
from tsfm import get_transform

from utils import * # create_logger, AverageMeter, seed_everything, check_cuda, save_checkpoint
import utils.config
import utils.checkpoint
import utils.metrics # TODO: for combined accuracy 


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
    log_filename = f'log_{config.setup.version}.txt'
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

    # criterion
    criterion = ArcFaceLoss() 

    # optimizer
    optimizer = get_optimizer(config, model.parameters())

    # lr_scheduler
    lr_scheduler = get_scheduler(config, optimizer)

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
    
        # SGDR
        if config.optimizer.name == 'cosine':
            lr_scheduler = get_scheduler(config, optimizer)
        # One cyclic lr
        elif config.optimizer.name == 'cyclic_lr':
            current_lr = lr_scheduler.get_lr()
            logger.info(current_lr[-1])                

        logger.info(train_logstr + valid_logstr)
    
        if combined_valid_accuracy >= best_model_accuracy:
            # torch.save(model.state_dict(), os.path.join(config.saved.model_dir, f"{config.setup.version}_best_model_{epoch}.pth"))
            best_model = model
            best_model_accuracy = combined_valid_accuracy        
            best_model_epoch = epoch
            # logger.info(f'A snapshot was saved to {filename}')

    logger.info(f'best score: {best_score:.3f}')

    print('Generating predictions...')

    submission, all_classes_preds = test_inference(test_loader, best_model)

    print('Number of unique sirnas', submission['predicted_sirna'].nunique())

    save_csv(config, submission, all_classes_preds)

    

def save_csv(config, submission, all_classes_preds):
    fn = f'{config.setup.version}_submission_{config.setup.cell_type}.csv'

    submission.to_csv(os.path.join(config.submission.submission_dir, fn), index=False)

    all_classes_preds['predicted_sirna'] = all_classes_preds['predicted_sirna'].apply(lambda o: o.numpy())

    fn = 'classes_' + fn

    all_classes_preds.to_csv(os.path.join(config.submission.submission_dir, fn), index=False)

    print('files saved to', config.submission.submission_dir)



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
    
    print('complete!')


if __name__ == "__main__":
    main()
