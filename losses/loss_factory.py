from typing import Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as edict

class ArcFaceLoss(nn.modules.Module):
    """"https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109"""

    def __init__(self,s=65.0,m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels, epoch=0):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        if torch.cuda.is_available():
            one_hot = torch.zeros(cosine.size(), device='cuda')
        else:
            one_hot = torch.zeros(cosine.size())

        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cosine, labels)
        gamma=1
        loss=(loss1+gamma*loss2)/(1+gamma)
        return loss


class AdMSoftmaxLoss(nn.Module):
    """https://github.com/cvqluu/Additive-Margin-Softmax-Loss-Pytorch"""

    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class FocalLoss(nn.Module):
    """https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/focal_loss.py"""

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


def cross_entropy() -> Any:
    return torch.nn.CrossEntropyLoss()

def binary_cross_entropy() -> Any:
    return torch.nn.BCEWithLogitsLoss()

def mse_loss() -> Any:
    return torch.nn.MSELoss()

def l1_loss() -> Any:
    return torch.nn.L1Loss()

def smooth_l1_loss() -> Any:
    return torch.nn.SmoothL1Loss()

def focal():
    return FocalLoss()

def arcface():
    return ArcFaceLoss()    

def amsoft(in_features, out_features):
    return AdMSoftmaxLoss(in_features, out_features, s=30.0, m=0.4)

def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)


if __name__ == "__main__":
    pass