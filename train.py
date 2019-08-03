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
import utils.metrics

def seed_everything():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)


def save_checkpoint(logger, state: Dict[str, Any], filename: str, model_dir: str) -> None:
    torch.save(state, os.path.join(model_dir, filename))
    logger.info(f'A snapshot was saved to {filename}')


def create_model(config):
    model = get_model(config)

    if config.setup.use_cuda and torch.cuda.device_count() > 1: 
        model = torch.nn.DataParallel(model)

    # use gpu
    if config.setup.use_cuda: 
        model = model.cuda()

    return model


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

    for i, data in tqdm(enumerate(train_loader), total=total_step):
        input_, id_codes, target = data

        # if using gpu
        if config.setup.use_cuda:
            input_, target = input_.cuda(), target.cuda()
        
        output = model(input_, target)
        
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


def validate_one_epoch(config, logger, val_loader, model, criterion):
    logger.info('validate()')

    model.eval()

    all_predicts, all_targets = [], []

    with torch.no_grad():
        batch_size = config.val.batch_size
        total_size = len(val_loader.dataset)
        total_step = math.ceil(total_size / batch_size)

        for i, data in enumerate(tqdm(data_loader, total==total_step)):
            input_, id_codes, target = data
        
            # if using gpu
            if config.setup.use_cuda:
                input_, target = input_.cuda(), target.cuda()
            
            output = model(input_, target)
            loss = criterion(output, target)
                        
            _, predicts = torch.max(output.cpu(), dim=1)
            all_predicts.append(predicts)    

            if target is not None:
                all_targets.append(target)

    predicts = torch.cat(all_predicts)
    targets = torch.cat(all_targets) if len(all_targets) else None

    val_score = utils.metrics.accuracy(predicts, targets)

    logger.info(f'{epoch} validation accuracy: {val_score}')
    return val_score


def run(config):
    # create logger
    log_filename = f'log_training_{config.setup.version}.txt'
    logger = create_logger(os.path.join(config.experiment_dir, log_filename))

    # seed
    np.random.seed(0)

    # get the directory to store models
    model_dir = config.experiment_dir

    # get dataloders
    train_loader, val_loader, test_loader = get_dataloader(config)

    logger.info('=' * 50)

    # valid_dl len: {len(val_loader)}
    logger.info(f'train_dl len: {len(train_loader)}')
    
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
        
        # start training
        train_score = train_one_epoch(config, logger, train_loader, model, epoch, criterion, optimizer, lr_scheduler)

        if config.setup.stage == 0:
            score = train_score
        else:
            score = validate_one_epoch(config, logger, val_loader, model, criterion)
        
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
    # run(config)

    print('complete!')


if __name__ == "__main__":
    main()