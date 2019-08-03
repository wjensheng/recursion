import logging
import pandas as pd
import os
import torch

from typing import *

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '../data'

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