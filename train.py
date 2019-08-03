from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
from tqdm import tqdm

import time
from typing import *

import numpy as np
import torch
import torch.nn.functional as F

from datasets import get_dataloader
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler

from utils import create_logger, AverageMeter
import utils.config
import utils.checkpoint
import utils.metrics # TODO: for combined accuracy 


def create_model(config):
    model = get_model(config)

    if config.setup.use_cuda and torch.cuda.device_count() > 1: 
        model = torch.nn.DataParallel(model)

    # use gpu
    if config.setup.use_cuda: 
        model = model.cuda()

    return model

def train_one_epoch(config, logger, train_loader, model, criterion, optimizer, num_grad_acc):
    logger.info('training')

    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()
    # train_momentum(model)

    num_steps = len(train_loader)

    end = time.time()

    for idx, data in enumerate(tqdm(train_loader)):
        input_, id_codes, target = data

        # if using gpu
        if config.setup.use_cuda:
            input_, target = input_.cuda(), target.cuda()
        
        output = model(input_, target)

        loss = criterion(output, target)
                
        _, predicts = torch.max(output.detach(), dim=1)
        
        losses.update(loss.data.item(), input_.size(0))

        loss.backward()
        
        if num_grad_acc is None:
            optimizer.step()
            optimizer.zero_grad()
        elif (idx+1) % num_grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()
            
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
    # eval_momentum(model)

    valid_fc_dict = defaultdict(list)

    num_steps = len(val_loader)

    with torch.no_grad():
        for idx, data in enumerate(tqdm(val_loader)):
            input_, id_codes, target = data            

            # if using gpu
            if config.setup.use_cuda:
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

    check_cuda(logger)

    # get the directory to store models
    model_dir = config.experiment_dir

    # get transformations setting
    train_tsfm = get_transform(config, 'train')
    test_tsfm = get_transform(config, 'test')

    # get dataloders
    train_loader, val_loader, test_loader = get_dataloader(config, train_tsfm, test_tsfm)

    # valid_dl len: {len(val_loader)}
    logger.info(f'train_dl len: {len(train_loader)}')
    logger.info(f'valid_dl len: {len(val_loader)}')
    
    logger.info('=' * 50)
    
    # model
    model = create_model(config)

    # optimizer, lr_scheduler, criterion
    optimizer = optimizer = get_optimizer(config, model.parameters())
    criterion = get_loss(config)
    lr_scheduler = get_scheduler(config, optimizer)

    last_epoch = 0
    best_score = 0.0
    best_epoch = 0

    for epoch in range(last_epoch + 1, config.train.num_epochs + 1):
        logger.info('-' * 50)
        
        train_loss = train_one_epoch(config, logger, train_loader, model, criterion, optimizer, config.train.num_grad_acc)
    
        train_logstr = (f'Epoch: {epoch}\t'
                        f'Train loss: {train_loss:.3f}\t')
    
        valid_loss, valid_accuracy = validate_one_epoch(config, logger, val_loader, model, criterion, valid_df)
    
        valid_logstr = (f'Val loss: {valid_loss:.3f}\t'
                        f'Val accuracy: {valid_accuracy:.3f}\t')
    
        lr_scheduler.step()
        # current_lr = lr_scheduler.get_lr()

        logger.info(train_logstr, valid_logstr)
    
        # save best score, model
        if valid_accuracy > best_score:
            best_score = valid_accuracy
            best_epoch = epoch

            data_to_save = {
                'epoch': epoch,
                'arch': config.model.arch,
                'state_dict': model.state_dict(),
                'best_score': best_score,
                'score': score,
                'optimizer': optimizer.state_dict(),
                'options': config
            }

            filename = config.setup.version 
            best_model_path = f'{filename}_e{epoch:02d}_{score:.04f}.pth'
            save_checkpoint(logger, data_to_save, best_model_path, model_dir)

    logger.info(f'best score: {best_score:.3f}')


        
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

    seed_everything()    
    run(config)

    print('complete!')


if __name__ == "__main__":
    main()