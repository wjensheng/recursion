from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .optimizer_factory import get_optimizer

def make_optimizer_with_center(config, model, center_criterion):
    optimizer = get_optimizer(config, model.parameters())
    optimizer_center = get_optimizer(config, center_criterion.parameters())
    return optimizer, optimizer_center