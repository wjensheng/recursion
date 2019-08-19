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


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class RecursionNet(nn.Module):

    def __init__(self, num_classes, neck_feat, in_planes=2048, model_name='se_resnet50'):
        super(RecursionNet, self).__init__()
                
        self.backbone = getattr(pretrainedmodels, model_name)(num_classes=1000)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck_feat = neck_feat
        self.in_planes = in_planes

        final_in_features = self.backbone.last_linear.in_features

        if 'se_resnet' in model_name:
            trained_kernel = self.backbone.layer0.conv1.weight
            self.backbone.layer1.conv1 = create_new_conv(trained_kernel)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        elif 'resnet' in model_name:
            self.in_planes = 512
            trained_kernel = self.backbone.conv1.weight            
            self.backbone.conv1 = create_new_conv(trained_kernel)

            before_downsample = list(list(self.backbone.children())[-3][0].children())[:-1]
            after_downsample =  list(list(self.backbone.children())[-3][1].children())
            last_block = nn.Sequential(*before_downsample, *after_downsample)

            self.backbone = nn.Sequential(*list(self.backbone.children())[:-3], last_block)
                
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        
    def forward(self, x):        
        global_feat = self.gap(self.backbone(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.size(0), -1)  # flatten to (bs, 2048)

        feat = self.bottleneck(global_feat)

        if self.training:
            print('training...')
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    # def extract_feat(self, x):
    #     x = self.backbone(x)
    #     x = self.pooling(x)
    #     x = self.flatten(x)        
    #     x = self.bn1(x)
    #     x = self.dropout1(x)
    #     x = self.fc1(x)
    #     x = self.relu(x)        
    #     x = self.bn2(x)
    #     x = self.dropout2(x)
    #     return x


def get_model(config):    
    num_classes = config.model.num_classes
    model_name = config.model.arch
    neck_feat = config.model.neck_feat
    in_planes = config.model.in_planes

    net = RecursionNet(num_classes=num_classes, neck_feat=neck_feat, 
                       in_planes=in_planes, model_name=model_name)
                  
    return net