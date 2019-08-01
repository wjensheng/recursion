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
from models import init_network
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler

import utils
from utils import create_logger, AverageMeter
import utils.config
import utils.checkpoint
import utils.metrics

def create_model(config, logger):
    logger.info(f'creating a model {config.model.arch}')

    model = get_custom_model(config)

    if config.setup.use_cuda and torch.cuda.device_count() > 1: 
        model = torch.nn.DataParallel(model)

    # use gpu
    if config.setup.use_cuda: 
        model = model.cuda()

    return model


def save_checkpoint(logger, state: Dict[str, Any], filename: str, model_dir: str) -> None:
    torch.save(state, os.path.join(model_dir, filename))
    logger.info(f'A snapshot was saved to {filename}')


def inference(config, data_loader, model):
    model.eval()

    all_predicts, all_targets = [], []

    with torch.no_grad():
        batch_size = config.val.batch_size
        total_size = len(data_loader.dataset)
        total_step = math.ceil(total_size / batch_size)

        for i, data in enumerate(tqdm(data_loader, total==total_size)):
            if data_loader.dataset.mode != 'test':
                input_, target = data
            else:
                input_, target = data, None

            # if using gpu
            if config.setup.use_cuda:
                input_, target = input_.cuda(), target.cuda()
            
            output = model(input_)            
            loss = criterion(output, target)
                        
            _, predicts = torch.max(output.cpu(), dim=1)
            all_predicts.append(predicts)    

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    targets = torch.cat(all_targets) if len(all_targets) else None

    return predicts, targets


def validate_one_epoch(logger, val_loader, model, epoch):
    logger.info('validate()')

    predicts, targets = inference(config, data_loader, model)
    val_score = utils.metrics.accuracy(predicts, targets)

    logger.info(f'{epoch} validation accuracy: {val_score}')
    return val_score


def train_one_epoch(config, logger, train_loader, model, epoch, criterion, optimizer, lr_scheduler):

    logger.info(f'epoch {epoch}')

    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_score = AverageMeter()

    model.train()

    num_steps = len(train_loader)

    end = time.time()
    lr_str = ''

    batch_size = config.val.batch_size
    total_size = len(train_loader.dataset)
    total_step = math.ceil(total_size / batch_size)

    for i, (input_, target) in tqdm(enumerate(train_loader), total=total_step):

        # if using gpu
        if config.setup.use_cuda:
            input_, target = input_.cuda(), target.cuda()
        
        # compute output
        output = model(input_)
        
        loss = criterion(output, target)

        # get metric
        _, predicts = torch.max(output.cpu(), dim=1)
        avg_score.update(utils.metrics.accuracy(predicts, target))

        # compute gradient and step
        losses.update(loss.data.item(), input_.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.train.log_freq == 0:
            logger.info(f'{epoch} [{i}/{num_steps}]\t'
                        f'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'accuracy {avg_score.val:.4f} ({avg_score.avg:.4f})'
                        + lr_str)

    logger.info(f' * accuracy on train {avg_score.avg:.4f}')

    return avg_score.avg


def run(config):
    # create logger
    log_filename = f'log_training_{config.setup.version}.txt'
    logger = create_logger(os.path.join(config.experiment_dir, log_filename))

    # seed
    np.random.seed(0)

    # get the directory to store models
    model_dir = config.experiment_dir

    logger.info('=' * 50)

    # get dataloders
    train_loader, val_loader, test_loader = get_dataloader(config)

    # valid_dl len: {len(val_loader)}
    logger.info(f'train_dl len: {len(train_loader)}')

    # get model    
    model = create_model(config, logger)
    
    # optimizer, lr_scheduler, criterion
    optimizer = get_optimizer(config, model.parameters())
    lr_scheduler = get_scheduler(config, optimizer)    
    criterion = get_loss(config)

    # # TODO: shift loading weights to here or create 
    # # another function to handle
    # # decide to train from scratch?
    # if args.weights is None:
    last_epoch = 0
    logger.info(f'training will start from epoch {last_epoch+1}')
    # else:
    #     last_checkpoint = torch.load(args.weights)
    
    best_score = 0.0
    best_epoch = 0

    # TODO: ignore gen_features for now

    # TODO: ignore gen_predict for now

    # best_model_path = args.weights

    for epoch in range(last_epoch + 1, config.train.num_epochs + 1):
        logger.info('-' * 50)
        
        # start training
        train_score = train_one_epoch(config, logger, train_loader, model, epoch, criterion, optimizer, lr_scheduler)

        if config.stage == 1:
            score = train_score
        else:
            score = validate_one_epoch(logger, val_loader, model, epoch)
        
        # save best score, model
        if score > best_score:
            best_score = score
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

            # TODO: what is config.version?
            filename = config.setup.version 
            best_model_path = f'{filename}_e{epoch:02d}_{score:.04f}.pth'
            save_checkpoint(logger, data_to_save, best_model_path, model_dir)

    logger.info(f'best score: {best_score:.04f}')


def parse_args():
    parser = argparse.ArgumentParser(description='RXRX')

    parser.add_argument('--config', help='model configuration file (YAML)', 
                        type=str, required=True)    
    # parser.add_argument('--weights', help='model to resume training', type=str)    
    # parser.add_argument('--gen_predict', help='make predictions for the testset and return', action='store_true')
    # parser.add_argument('--gen_features', help='calculate features for the given set', action='store_true')
    # parser.add_argument('--summary', help='show model summary', action='store_true')
    # parser.add_argument('--num_ttas', help='override number of TTAs', type=int, default=0)
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = utils.config.load_config(args.config, args)

    pprint.PrettyPrinter(indent=2).pprint(config)

    if not os.path.exists(config.experiment_dir):
        os.makedirs(config.experiment_dir)    

    run(config)

    print('complete!')


if __name__ == "__main__":
    model_params = {}
    model_params['architecture'] = 'resnet18'
    model_params['pooling'] = 'gem'
    model_params['local_whitening'] = False
    model_params['regional'] = False
    model_params['whitening'] = False
    # model_params['mean'] = ...  # will use default
    # model_params['std'] = ...  # will use default
    model_params['pretrained'] = True
    model = init_network(model_params)
    print(model)