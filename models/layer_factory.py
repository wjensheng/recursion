import torch
import torch.nn as nn

import layers.functional as LF
from layers.loss import ContrastiveLoss

import easydict as edict
from typing import *

# TODO: HOW TO CONVERT?