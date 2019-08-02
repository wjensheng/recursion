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
from models import init_network, extract_vectors, get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
from layers.loss import ContrastiveLoss

import utils
from utils import create_logger, AverageMeter
import utils.config
import utils.checkpoint
import utils.metrics

def seed_everything():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)


def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)


def create_model(config):
    model = get_model(config)

    if config.setup.use_cuda and torch.cuda.device_count() > 1: 
        model = torch.nn.DataParallel(model)

    # use gpu
    if config.setup.use_cuda: 
        model = model.cuda()

    return model


def get_criterion(config):
    if config.loss.name == 'contrastive':
        criterion = ContrastiveLoss(margin=0.3)
    else:
        criterion = get_loss(config)

    if config.setup.use_cuda:
        criterion = criterion.cuda()

    return criterion


def get_model_params(config, model):
    # parameters split into features, pool, whitening 
    # IMPORTANT: no weight decay for pooling parameter p in GeM or regional-GeM
    parameters = []

    # add feature parameters
    parameters.append({'params': model.features.parameters()})

    # add local whitening if exists
    if model.lwhiten is not None:
        parameters.append({'params': model.lwhiten.parameters()})

    # add pooling parameters (or regional whitening which is part of the pooling layer!)
    if not config.model.regional:
        # global, only pooling parameter p weight decay should be 0
        parameters.append({'params': model.pool.parameters(), 'lr': config.model.lr*10, 'weight_decay': 0})
    else:
        # regional, pooling parameter p weight decay should be 0, 
        # and we want to add regional whitening if it is there
        parameters.append({'params': model.pool.rpool.parameters(), 'lr': config.model.lr*10, 'weight_decay': 0})
        if model.pool.whiten is not None:
            parameters.append({'params': model.pool.whiten.parameters()})

    # add final whitening if exists
    if model.whiten is not None:
        parameters.append({'params': model.whiten.parameters()})

    return parameters


def train(config, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()


    batch_size = config.val.batch_size
    total_size = len(train_loader.dataset)
    total_step = math.ceil(total_size / batch_size)

    for i, data in tqdm(enumerate(train_loader), total=total_step):
        input, id_codes, target = data
        
        # if using gpu
        if config.setup.use_cuda:
            input, target = input.cuda(), target.cuda()
        
        optimizer.zero_grad()

        output = model(input).squeeze()

        print(output.size())

        loss = criterion(output, target)
        losses.update(loss.item(), input.size(0))
        loss.backward()

        # do one step for multiple batches
        # accumulated gradients are used
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % config.train.log_freq == 0 or i == 0 or (i+1) == len(train_loader):
            print('>> Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i+1, len(train_loader), batch_time=batch_time, loss=losses))

    return losses.avg


def validate(config, val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    end = time.time()
    for i, data in enumerate(val_loader):
        input, id_codes, target = data

        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple
        
        if config.setup.use_cuda: output = torch.zeros(model.meta['outputdim'], nq*ni).cuda()
        else: output = torch.zeros(model.meta['outputdim'], nq*ni)

        for q in range(nq):
            for imi in range(ni):

                # compute output vector for image imi of query q
                if config.setup.use_cuda: output[:, q*ni + imi] = model(input[q][imi].cuda()).squeeze()
                else: output[:, q*ni + imi] = model(input[q][imi]).squeeze()

        
        if config.setup.use_cuda: loss = criterion(output, torch.cat(target).cuda())
        else: loss = criterion(output, torch.cat(target))
        
        # record loss
        losses.update(loss.item()/nq, nq)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % config.train.log_freq == 0 or i == 0 or (i+1) == len(val_loader):
            print('>> Val: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch+1, i+1, len(val_loader), batch_time=batch_time, loss=losses))

    return losses.avg


def run(config):
    # get dataloders
    train_loader, val_loader, test_loader = get_dataloader(config)

    # model
    model = create_model(config)

    # optimizer, lr_scheduler, criterion
    optimizer = get_optimizer(config, get_model_params(config, model))
    lr_scheduler = get_scheduler(config, optimizer)    
    criterion = get_criterion(config)

    start_epoch = 0

    for epoch in range(start_epoch, config.train.num_epochs):
        # train for one epoch on train set
        loss = train(config, train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        loss = validate(config, val_loader, model, criterion, epoch)

        # adjust learning rate for each epoch
        scheduler.step()

        # remember best loss and save checkpoint
        is_best = loss < min_loss
        min_loss = min(loss, min_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'meta': model.meta,
            'state_dict': model.state_dict(),
            'min_loss': min_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, experiment_dir.directory)

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

    seed_everything()
    run(config)

    print('complete!')


if __name__ == "__main__":
    main()

    # nq = len(input) # number of training tuples
    #     ni = len(input[0]) # number of images per tuple

    #     for q in range(nq):
    #         if config.setup.use_cuda: output = torch.zeros(model.meta['outputdim'], ni).cuda()
    #         else: output = torch.zeros(model.meta['outputdim'], ni)

    #         for imi in range(ni):

    #             print('output shape:', output.size())
    #             print('input[q][imi]', input[q][imi].size())

    #              # compute output vector for image imi
    #             if config.setup.use_cuda: output[:, imi] = model(input[q][imi].cuda()).squeeze()
    #             else: output[:, imi] = model(input[q][imi]).squeeze()

    #         if config.setup.use_cuda: loss = criterion(output, target[q].cuda())
    #         else: loss = criterion(output, target[q])

    #         losses.update(loss.item())
    #         loss.backward()    