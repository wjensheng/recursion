import pprint
import os
import argparse
import torch
import yaml

from easydict import EasyDict as edict
from typing import *

def _get_default_config(filename: str, args: Any) -> edict:
    cfg = edict()

    # experiments
    cfg.experiment_dir = 'experiments/'    

    # setups
    cfg.setup = edict()
    cfg.setup.use_cuda = False
    cfg.setup.cell_type = -1
    cfg.setup.stage = 1
    cfg.setup.combine = True
    cfg.setup.version = 'first_attempt'
    cfg.setup.use_small = False
    cfg.setup.test_size = 0.1
    cfg.setup.val_set = False

    # saved model
    cfg.saved = edict()
    cfg.saved.model_dir = 'experiments/models'
    # cfg.saved.pth_fn = args.pth_fn
    
    # data
    cfg.data = edict()
    cfg.data.data_dir = 'data/'
    # cfg.data.params = edict()

    # augmentations
    # cfg.augmentations = edict()
    # cfg.augmentations.blur = 0
    # cfg.augmentations.color = 0

    # num works
    cfg.num_workers = 2

    # model
    cfg.model = edict()
    cfg.model.arch = 'resnet34'
    cfg.model.image_size = 512 # TODO: do not resize
    cfg.model.input_size = 512
    cfg.model.num_classes = 1108
    
    # train
    cfg.train = edict()
    cfg.train.batch_size = 16 # * torch.cuda.device_count()
    cfg.train.num_epochs = 40
    cfg.train.log_freq = 100
    cfg.train.lr_scheduler = None
    # cfg.train.num_ttas = 1
    
    # valid
    cfg.val = edict()
    cfg.val.batch_size = 16 # * torch.cuda.device_count()
    # cfg.test.num_ttas = 1

    # test
    cfg.test = edict()
    cfg.test.batch_size = 16 # * torch.cuda.device_count()
    cfg.test.num_ttas = 1

    # optimizer
    cfg.optimizer = edict()
    cfg.optimizer.name = 'adam'
    cfg.optimizer.params = edict()

    # scheduler
    cfg.scheduler = edict()
    cfg.scheduler.name = ''
    cfg.scheduler.params = edict()

    # loss
    cfg.loss = edict()
    cfg.loss.name = 'none'

    return cfg

def _merge_config(src: edict, dst: edict) -> edict:
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v

def load_config(config_path: str, args: Any) -> edict:
    with open(config_path) as f:
        yaml_config = edict(yaml.load(f, Loader=yaml.SafeLoader))

    config = _get_default_config(config_path, args)
    _merge_config(yaml_config, config)

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_type', help='random: [-1], cell_type: [0, 1, 2, 3]', type=int, default=-1)
    # parser.add_argument('--config', help='model configuration file (YAML)', type=str, required=True)
    args = parser.parse_args()

    # TODO: merge args and config
    config = load_config('configs/resnet18.yml', args)

    pprint.PrettyPrinter(indent=2).pprint(config)