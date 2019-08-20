import os
import argparse

import torch
import torch.nn as nn

from collections import defaultdict
from fastprogress import master_bar, progress_bar

from datasets import get_dataframes, get_datasets, make_data_loader
from models import get_model
from losses import make_loss_with_center
from optimizers import make_optimizer_with_center
from schedulers import get_scheduler

from utils import *
import utils.config

import wandb


def train_one_epoch(config, train_loader, model, center_criterion, optimizer, optimizer_center, scheduler, loss_fn, center_loss_weight, mb=None):
    
    model.train()

    losses = utils.AverageMeter()

    optimizer.zero_grad()
    optimizer_center.zero_grad()

    for idx, data in enumerate(progress_bar(train_loader, parent=mb)):
        img, id_codes, target = data

        # if using gpu
        if torch.cuda.is_available():
            img = img.cuda()
            target = target.cuda()

        score, feat = model(img)

        loss = loss_fn(score, feat, target)

        losses.update(loss.data.item(), img.size(0))
        loss.backward()

        optimizer.step()

        for param in center_criterion.parameters():
            param.grad.data *= (1. / center_loss_weight)

        optimizer_center.step()

        scheduler.step()

    return losses.avg


def validate_one_epoch(config, val_loader, model, loss_fn, valid_df, mb=None):
    
    model.eval()
    
    losses = utils.AverageMeter()

    valid_fc_dict = defaultdict(list)

    with torch.no_grad():
        for idx, data in enumerate(progress_bar(val_loader, parent=mb)):
            img, id_codes, target = data

            # if using gpu
            if torch.cuda.is_available():
                img, target = img.cuda(), target.cuda()
                        
            feat = model(img)
            loss = loss_fn(score, feat, target) # TODO: what is score?
                        
            losses.update(loss.data.item(), img.size(0))
            
            for i in range(len(id_codes)):
                valid_fc_dict[id_codes[i]] += output[i],
                
    combined_valid_accuracy = utils.metrics.combined_accuracy(valid_fc_dict, valid_df)

    return losses.avg, combined_valid_accuracy
    

def create_model(config):
    model = get_model(config)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1: 
        model = torch.nn.DataParallel(model)

    # use gpu
    if torch.cuda.is_available(): 
        model = model.cuda()

    return model


def do_train_with_center(config, model, center_criterion, train_loader, val_loader, valid_df, optimizer, optimizer_center, scheduler, loss_fn, start_epoch):

    best_score = 0.0
    best_epoch = 0
    best_model = None    

    mb = master_bar(range(start_epoch, config.train.num_epochs))

    for epoch in mb:        
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        train_loss = train_one_epoch(config, train_loader, model, center_criterion, optimizer, optimizer_center, scheduler, loss_fn, config.loss.params.CENTER_LOSS_WEIGHT, mb)

        val_loss, val_accuracy = validate_one_epoch(config, val_loader, model, loss_fn, valid_df, mb)
    
        scheduler = get_scheduler(config, optimizer)
        
        # wandb.log({
        #     'Train loss': train_loss,
        #     'Valid loss': val_loss,
        #     'Valid accuracy': val_accuracy
        # })

        print(train_loss, val_loss, val_accuracy)
    
        # save best score, model
        if val_accuracy > best_score:
            best_score = val_accuracy
            best_epoch = epoch
            best_model = model
                    
    # checkpoint = f't{config.setup.cell_type}_e{best_epoch:02d}_{best_score:.04f}.pth'
    # torch.save(best_model.state_dict(), os.path.join(config.saved.model_dir, checkpoint))

    return best_model


def run(config):

    _, valid_df, _ = get_dataframes(config)

    train_loader, val_loader = make_data_loader(config)

    model = create_model(config)

    loss_func, center_criterion = make_loss_with_center(config) 
    optimizer, optimizer_center = make_optimizer_with_center(config, model, center_criterion)
    scheduler = get_scheduler(config, optimizer)

    start_epoch = 0

    do_train_with_center(
        config, 
        model, 
        center_criterion, 
        train_loader, 
        val_loader, 
        valid_df, 
        optimizer, 
        optimizer_center, 
        scheduler, 
        loss_func, 
        start_epoch
    )

def test_model(config):
    model = create_model(config)
    loss_func, center_criterion = make_loss_with_center(config)

    print(model)
    print(loss_func)
    print(center_criterion)

    print('\ntraining...\n')
    model.train()

    img = torch.randn((16, 6, 224, 224))
    target = torch.tensor([1, 2, 3, 4] * 4)

    score, feat = model(img)
    loss = loss_func(score, feat, target)
    
    print('score size:', score.size())
    print('feature size:', feat.size())
    print('loss:', loss.item())

    print('\nvalidation...\n')
    model.eval()

    feat = model(img)

    print(feat.size())


def test_dl(config):
    train_loader, val_loader = make_data_loader(config)

    print(train_loader)

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

    # test_model(config)
    test_dl(config)
    # run(config)

    print('complete!')


if __name__ == "__main__":
    main()    