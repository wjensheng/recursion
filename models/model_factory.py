import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels
import easydict as edict

import models.pooling as pooling
from models.metric_learning import *

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
    new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():
        new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)

    return new_conv


class RecursionNet(nn.Module):

    def __init__(self, n_classes, model_name='resnet50', 
                 fc_dim=512, loss_module='softmax'):
        super(RecursionNet, self).__init__()        
                
        self.backbone = getattr(pretrainedmodels, model_name)(num_classes=1000)

        final_in_features = self.backbone.last_linear.in_features        

        if 'se_resnet' in model_name:
            trained_kernel = self.backbone.layer0.conv1.weight
            self.backbone.layer1.conv1 = create_new_conv(trained_kernel)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            self.expand = 1
        
        elif 'resnet' in model_name:
            trained_kernel = self.backbone.conv1.weight            
            self.backbone.conv1 = create_new_conv(trained_kernel)

            before_downsample = list(list(self.backbone.children())[-3][0].children())[:-1]
            after_downsample =  list(list(self.backbone.children())[-3][1].children())
            last_block = nn.Sequential(*before_downsample, *after_downsample)

            self.backbone = nn.Sequential(*list(self.backbone.children())[:-3], last_block)
            self.expand = 1
        
        elif 'densenet' in model_name:
            trained_kernel = self.backbone.features.conv0.weight            
            self.backbone.features.conv0 = create_new_conv(trained_kernel)
            self.backbone = nn.Sequential(*list(self.backbone.features)[:-1])
            self.expand = 2           

        else:
            raise ValueError('Wrong model_name')
        
        self.pooling = AdaptiveConcatPool2d()
        self.flatten = Flatten()        
        self.bn1 = nn.BatchNorm1d(1024 * self.expand)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(1024 * self.expand, 512 * self.expand)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512 * self.expand)   
        self.dropout2 = nn.Dropout(p=0.5)     
        self._init_params()        
    
        final_in_features = fc_dim * self.expand
        
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes)
        elif loss_module == 'sphereface':
            self.final = SphereProduct(final_in_features, n_classes)
        elif loss_module == 'amsoftmax':
            self.final = AdaptiveMargin(final_in_features, n_classes)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

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
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)        
        x = self.bn2(x)
        x = self.dropout2(x)
        return x


def get_model(config):
    n_classes = config.model.num_classes
    model_name = config.model.arch
    fc_dim = config.model.fc_dim
    loss_module = config.loss.name

    net = RecursionNet(n_classes=n_classes, model_name=model_name,
                       fc_dim=fc_dim, loss_module=loss_module)
                  
    return net