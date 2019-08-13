from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.optim as optim

from easydict import EasyDict as edict

# from models import get_pretrainedmodels


def adam(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0,
         amsgrad=False, **_):
    if isinstance(betas, str):
        betas = eval(betas)
    # print('weight decay:', weight_decay)
    return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay,
                      amsgrad=amsgrad)


def adamw(parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, 
          amsgrad=False, **_):
    return optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay, 
                       amsgrad=amsgrad)

def sgd(parameters, lr=0.001, momentum=0.9, weight_decay=0, nesterov=True, **_):
    return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,
                     nesterov=nesterov)


def get_optimizer(config, parameters):
    f = globals().get(config.optimizer.name)
    return f(parameters, **config.optimizer.params)

def test():
    c = edict()
    # optimizer
    c.optimizer = edict()
    c.optimizer.name = 'adam'
    c.optimizer.params = edict()

    # model = get_pretrainedmodels(num_outputs=10)

    # optim = get_optimizer(c, model.parameters())

    # print(optim)

if __name__ == "__main__":
    test()