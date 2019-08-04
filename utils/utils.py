import logging
import pandas as pd
import numpy as np
import os
import torch

from typing import *

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '../data'

def seed_everything():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)


def remove_redundant_keys(state_dict: OrderedDict):
    # remove DataParallel wrapping
    if 'module' in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # str.replace() can't be used because of unintended key removal (e.g. se-module)
                new_state_dict[k[7:]] = v
    else:
        new_state_dict = state_dict

    return new_state_dict


def save_checkpoint(model_dir, filename, model, epoch, best_score, optimizer=None, save_arch=False, params=None):
    attributes = {
        'epoch': epoch,
        'state_dict': remove_redundant_keys(model.state_dict()),
        'best_score': best_score
    }

    if optimizer is not None:
        attributes['optimizer'] = optimizer.state_dict()

    if save_arch:
        attributes['arch'] = model

    if params is not None:
        attributes['params'] = params

    try:
        torch.save(attributes, os.path.join(model_dir, filename))
        
    except TypeError:
        if 'arch' in attributes:
            print('Model architecture will be ignored because the architecture includes non-pickable objects.')
            del attributes['arch']
            torch.save(attributes, os.path.join(model_dir, filename))    


def load_checkpoint(path, model=None, optimizer=None, params=False, epoch=False):
    resume = torch.load(path)
    rets = dict()

    if model is not None:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(remove_redundant_keys(resume['state_dict']))
        else:
            model.load_state_dict(remove_redundant_keys(resume['state_dict']))

        rets['model'] = model

    if optimizer is not None:
        optimizer.load_state_dict(resume['optimizer'])
        rets['optimizer'] = optimizer
    if params:
        rets['params'] = resume['params']
    if epoch:
        rets['epoch'] = resume['epoch']

    return rets


def load_model(path, is_inference=True):
    resume = torch.load(path)
    model = resume['arch']
    model.load_state_dict(resume['state_dict'])
    if is_inference:
        model.eval()
    return model


def check_cuda(logger):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f'{num_gpus} gpu(s) available!')
    else:
        logger.info('Using cpu!')
    

def create_logger(filename: str) -> Any:
    logger_name = 'logger'
    file_fmt_str = '%(asctime)s %(message)s'
    console_fmt_str = '%(message)s'
    file_level = logging.DEBUG

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_fmt = logging.Formatter(file_fmt_str, '%m-%d %H:%M:%S')
    log_file = logging.FileHandler(filename)
    log_file.setLevel(file_level)
    log_file.setFormatter(file_fmt)
    logger.addHandler(log_file)

    console_fmt = logging.Formatter(console_fmt_str)
    log_console = logging.StreamHandler()
    log_console.setLevel(logging.DEBUG)
    log_console.setFormatter(console_fmt)
    logger.addHandler(log_console)

    return logger

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count        

        
def test():
    train = pd.read_csv(os.path.join(DATA_DIR, 'full_train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    hepg2_train = filter_experiments(train, CELL_TYPE[0])
    hepg2_test = filter_experiments(test, CELL_TYPE[0])

    
    t, v = train_val_exp_split(hepg2_train, hepg2_test)

    print(t['experiment'].value_counts())
    print(v['experiment'].value_counts())


if __name__ == "__main__":
    test()