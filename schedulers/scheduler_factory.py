import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torch.optim import Optimizer

def step(optimizer, last_epoch, step_size=10, gamma=0.1, **_):
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch)

def multi_step(optimizer, last_epoch, milestones=[500, 5000], gamma=0.1, **_):
    if isinstance(milestones, str):
        milestones = eval(milestones)
    return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma,
                                    last_epoch=last_epoch)

def exponential(optimizer, last_epoch, gamma=0.995, **_):
    return lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)

def none(optimizer, last_epoch, **_):
    return lr_scheduler.StepLR(optimizer, step_size=10000000, last_epoch=last_epoch)

def reduce_lr_on_plateau(optimizer, last_epoch, mode='max', factor=0.1,
                         patience=10, threshold=0.0001, threshold_mode='rel',
                         cooldown=0, min_lr=0, **_):

    return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor,
                                          patience=patience, threshold=threshold,
                                          threshold_mode=threshold_mode,
                                          cooldown=cooldown, min_lr=min_lr)

def cosine(optimizer, last_epoch, T_max=50, eta_min=0.00001, **_):
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min,
                                          last_epoch=last_epoch)


class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


def cyclic_lr(optimizer, last_epoch, base_lr=0.001, max_lr=0.01,
              step_size_up=2000, step_size_down=None, mode='triangular',
              gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True,
              base_momentum=0.8, max_momentum=0.9, **_):
    return CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                    step_size_up=step_size_up, step_size_down=
                    step_size_down, mode=mode, gamma=gamma,
                    scale_mode=scale_mode, cycle_momentum=
                    cycle_momentum, base_momentum=base_momentum,
                    max_momentum=max_momentum, last_epoch=last_epoch)


        

def get_scheduler(config, optimizer, last_epoch=-1):
    func = globals().get(config.scheduler.name)
    return func(optimizer, last_epoch, **config.scheduler.params)
