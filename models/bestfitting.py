import os
import math
import torch.functional as F
import torch
import torch.nn as nn
from torchvision import models

# def pretrained_model(config, num_classes):
#     m = models.resnet34(pretrained=False, num_classes=num_classes)    
    
#     new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

#     m.conv1 = new_conv
    
#     m.load_state_dict(torch.load(os.path.join(config.saved.model_dir, 
#                                               config.saved.model)))
    
#     return nn.Sequential(*list(m.children())[:-2])


# class ArcMarginProduct(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(ArcMarginProduct, self).__init__()
#         self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)

#     def forward(self, features):
#         cosine = F.linear(F.normalize(features), F.normalize(self.weight.cuda()))
#         return cosine


# class BestFittingModel(nn.Module):
#     def __init__(self, config, num_classes, extract_feature=False):
#         super(BestFittingModel, self).__init__()
#         self.model = pretrained_model(config, num_classes)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.arc_margin_product = ArcMarginProduct(512, num_classes)
#         self.EX = 1
#         self.bn1 = nn.BatchNorm1d(1024 * self.EX)
#         self.fc1 = nn.Linear(1024 * self.EX, 512 * self.EX)
#         self.bn2 = nn.BatchNorm1d(512 * self.EX)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(512 * self.EX, 512)
#         self.bn3 = nn.BatchNorm1d(512)
#         self.extract_feature = extract_feature

#     def forward(self, x):
#         e5 = self.model(x)
#         x = torch.cat((nn.AdaptiveAvgPool2d(1)(e5), nn.AdaptiveMaxPool2d(1)(e5)), dim=1)
#         x = x.view(x.size(0), -1)
#         x = self.bn1(x)
#         x = F.dropout(x, p=0.25)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.bn2(x)
#         x = F.dropout(x, p=0.5)

#         x = x.view(x.size(0), -1)

#         x = self.fc2(x)
#         feature = self.bn3(x)

#         cosine = self.arc_margin_product(feature)
#         if self.extract_feature:
#             return cosine, feature
#         else:
#             return cosine


# class ArcFaceLoss(nn.modules.Module):
#     def __init__(self,s=65.0,m=0.5):
#         super(ArcFaceLoss, self).__init__()
#         self.classify_loss = nn.CrossEntropyLoss()
#         self.s = s
#         self.easy_margin = False
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m

#     def forward(self, logits, labels, epoch=0):
#         cosine = logits
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         phi = cosine * self.cos_m - sine * self.sin_m
#         if self.easy_margin:
#             phi = torch.where(cosine > 0, phi, cosine)
#         else:
#             phi = torch.where(cosine > self.th, phi, cosine - self.mm)

#         one_hot = torch.zeros(cosine.size(), device='cuda')
#         one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
#         output *= self.s
#         loss1 = self.classify_loss(output, labels)
#         loss2 = self.classify_loss(cosine, labels)
#         gamma=1
#         loss=(loss1+gamma*loss2)/(1+gamma)
#         return loss