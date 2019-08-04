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
    cfg.setup.use_cuda = True
    cfg.setup.cell_type = 3
    cfg.setup.stage = 1
    cfg.setup.version = 'first_attempt'
    cfg.setup.is_control = False

    # saved model
    cfg.saved = edict()
    cfg.saved.model_dir = 'experiments/models'
    
    # data
    cfg.data = edict()
    cfg.data.data_dir = 'data/'
    cfg.data.train = 'train.csv' # 'U2OS_train.csv'
    cfg.data.test = 'test.csv'

    # # transforms
    # cfg.transform = edict()
    # cfg.transform.train = 'default_train'
    # cfg.transform.test = 'default_test'
    # cfg.transform.num_preprocessor = 4
    # cfg.transform.params = edict()

    # num works
    cfg.num_workers = 4 # * torch.cuda.device_count()

    # model
    cfg.model = edict()
    cfg.model.arch = 'resnet18'
    cfg.model.pool = 'GeM' # 'SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling'
    cfg.model.local_whitening = False
    cfg.model.use_fc = False
    cfg.model.fc_dim = 512
    cfg.model.dropout = 0
    cfg.model.loss_module = 'arcface' # 'arcface', 'cosface', 'softmax'
    cfg.model.s = 30.0
    cfg.model.margin = 0.5
    cfg.model.regional = False
    cfg.model.whitening = False
    cfg.model.image_size = 512 # resize
    cfg.model.num_classes = 1108
    cfg.model.pretrained = True
    cfg.model.lr = 3e-4
    
    # train
    cfg.train = edict()
    cfg.train.batch_size = 64 # * torch.cuda.device_count()
    cfg.train.num_grad_acc = None # 2?
    cfg.train.num_epochs = 50
    cfg.train.log_freq = 100
    # cfg.train.num_ttas = 1
    
    # valid
    cfg.val = edict()
    cfg.val.batch_size = 64 # * torch.cuda.device_count()
    cfg.val.log_freq = 100
    # cfg.test.num_ttas = 1

    # test
    cfg.test = edict()
    cfg.test.batch_size = 64 # * torch.cuda.device_count()
    # cfg.test.num_ttas = 1

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