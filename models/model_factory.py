import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import pretrainedmodels
from pytorchcv.model_provider import get_model

from easydict import EasyDict as edict

# TODO
# -experiment various head

# def create_head(model, in_features, num_classes):
#     return nn.Sequential(*list(model.children())[:-2]), create_head(4096, classes)

class DnModel(nn.Module):
    def __init__(self, num_classes=1108) -> None:
        super().__init__()
        self.num_classes = num_classes

        # self.model = pretrainedmodels.__dict__[config.model.arch](num_classes=1000, pretrained='imagenet')

        self.model = get_model('densenet121', pretrained=True)

        # modify tail
        trained_kernel = self.model.features.init_block.conv.weight

        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)

        self.model.features.init_block.conv = new_conv

        # modify head
        self.model.features[-1] = nn.AdaptiveAvgPool2d(1)
        in_features = self.model.output.in_features
        
        self.model.output = nn.Linear(in_features, self.num_classes)        

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.features(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.forward(images)


class RnModel(nn.Module):
    def __init__(self, num_classes=1108) -> None:
        super().__init__()
        self.num_classes = num_classes

        # self.model = pretrainedmodels.__dict__[config.model.arch](num_classes=1000, pretrained='imagenet')

        self.model = get_model('resnet34', pretrained=True)

        # modify tail
        trained_kernel = self.model.features.init_block.conv.conv.weight

        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)

        self.model.features.init_block.conv.conv = new_conv

        # modify head
        self.model.features[-1] = nn.AdaptiveAvgPool2d(1)
        in_features = self.model.output.in_features
        
        self.model.output = nn.Linear(in_features, self.num_classes)        

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.features(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.forward(images)

def get_resnet34():
    return RnModel()

def get_densenet121():
    return DnModel()    

def get_custom_model(config):
    f = globals().get('get_' + config.model.arch)
    return f()

if __name__ == '__main__':
    config = edict()
    config.model = edict()
    config.model.arch = 'densenet121'
    config.model.num_classes = 1108
    model = get_custom_model(config)

    # print(get_custom_model(config))