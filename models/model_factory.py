import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels
import easydict as edict
import numpy as np

import models.pooling as pooling
from models.metric_learning import *
from models.resnet import *
from models.densenet import *

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x.view(x.size(0), -1)        


def create_new_conv(trained_kernel):
    trained_kernel_weights = trained_kernel.weight    

    new_conv = nn.Conv2d(in_channels=6, 
                         out_channels=trained_kernel.out_channels, 
                         kernel_size=trained_kernel.kernel_size, 
                         stride=trained_kernel.stride, 
                         padding=trained_kernel.padding, 
                         bias=False)

    with torch.no_grad():
        new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel_weights, 1)]*6, dim=1)

    return new_conv

class RecursionNet(nn.Module):

    def __init__(self, num_classes, model_name='resnet18', 
                 fc_dim=512, loss_module='softmax', antialias=True, filter_size=5):
        super(RecursionNet, self).__init__()            
        
        if antialias: # only supports resnet18's weights
            self.backbone = globals().get(model_name)(filter_size=filter_size)
            if torch.cuda.is_available():                
                self.backbone.load_state_dict(torch.load('weights/{0}_lpf{1}.pth.tar'.format(model_name, filter_size))['state_dict'])
            else:
                self.backbone.load_state_dict(torch.load('weights/{0}_lpf{1}.pth.tar'.format(model_name, filter_size), map_location=torch.device('cpu'))['state_dict'])

            # change first filter
            trained_kernel = self.backbone.conv1
            self.backbone.conv1 = create_new_conv(trained_kernel)

            # get in_features
            final_in_features = self.backbone.fc.in_features            

            # remove head
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            
        else:
            # self.backbone = getattr(pretrainedmodels, model_name)(num_classes=1000)
            self.backbone = globals().get(model_name)(filter_size=filter_size, pretrained=True)

            trained_kernel = self.backbone.features.conv0            
            self.backbone.features.conv0 = create_new_conv(trained_kernel)

            final_in_features = self.backbone.last_linear.in_features

            self.backbone = nn.Sequential(*list(self.backbone.features)[:-1])            

                        
        self.pooling = AdaptiveConcatPool2d()
        self.flatten = Flatten()        
        self.bn1 = nn.BatchNorm1d(final_in_features * 2)
        self.fc1 = nn.Linear(final_in_features * 2, final_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(final_in_features)   
        self.dropout1 = nn.Dropout(p=0.25)
        self._init_params()  

        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, num_classes)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, num_classes)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, num_classes)
        elif loss_module == 'sphereface':
            self.final = SphereProduct(final_in_features, num_classes)
        elif loss_module == 'amsoftmax':
            self.final = AdaptiveMargin(final_in_features, num_classes)
        else:
            self.final = nn.Linear(final_in_features, num_classes)

    def _init_params(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
        
    def forward(self, x):        
        feature = self.extract_feat(x)
        logits = self.final(feature)
        return logits

    def extract_feat(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        x = self.flatten(x)        
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.relu(x)        
        x = self.bn2(x)
        x = self.dropout1(x)
        return x

def get_model(config):
    num_classes = config.model.num_classes
    model_name = config.model.arch
    fc_dim = config.model.fc_dim
    loss_module = config.loss.name
    antialias = config.model.antialias

    net = RecursionNet(num_classes=num_classes, model_name=config.model.arch,
                       fc_dim=fc_dim, loss_module=loss_module, antialias=antialias)
                  
    return net