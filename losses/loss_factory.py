from typing import Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as edict

class ArcFaceLoss(nn.modules.Module):
    """"https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109"""

    def __init__(self, s=65.0, m=0.5, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        # self.ls_eps = ls_eps  # label smoothing

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

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 's=' + str(self.s) \
               + ', m=' + str(self.m) \
               + ', easy_margin=' + str(self.easy_margin) + ')'


class CosFaceLoss(nn.modules.Module):

    def __init__(self, s=30.0, m=0.40):
        super(CosFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()        
        self.s = s
        self.m = m

    def forward(self, logits, labels):
        cosine = logits
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        if torch.cuda.is_available():
            one_hot = torch.zeros(cosine.size(), device='cuda')
        else:
            one_hot = torch.zeros(cosine.size())

        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cosine, labels)
        gamma=1
        loss=(loss1+gamma*loss2)/(1+gamma)
        return loss
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 's=' + str(self.s) \
               + ', m=' + str(self.m) + ')'



class AdaCosLoss(nn.modules.Module):
    
    def __init__(self, in_features, out_features, m=0.50, ls_eps=0, theta_zero=math.pi/4):
        super(AdaCosLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.theta_zero = theta_zero
        self.s = math.log(out_features - 1) / math.cos(theta_zero)
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        

    def forward(self, logits, labels):
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta)
            self.s = torch.log(B_avg) / torch.cos(torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med))
        output *= self.s
        # return output

        loss = self.classify_loss(output, labels)

        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'm=' + str(self.m) \
               + ', in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', ls_eps' + str(self.ls_eps) + ')'



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

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 's=' + str(self.s) \
               + ', m=' + str(self.m) \
               + ', in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'


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

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'gamma=' + str(self.gamma) \
               + ', ce=' + str(self.ce) + ')'


class LabelSmoothingCrossEntropy(nn.Module):
    """https://github.com/fastai/fastai/blob/221e4aae0304ef5d32c9d3645afade6751f074f0/fastai/layers.py"""

    def __init__(self, eps:float=0.1, reduction='mean'): 
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps,self.reduction = eps,reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum': loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


def cross_entropy(**_):
    return torch.nn.CrossEntropyLoss()

def focal(**_):
    return FocalLoss(**_)

def arcface(**_):
    return ArcFaceLoss(**_)    

def cosface(**_):
    return CosFaceLoss(**_)    

def adacos(in_features, out_features, **_):
    return AdaCosLoss(in_features, out_features, m=0.50, ls_eps=0, theta_zero=math.pi/4)

def amsoftmax(in_features, out_features, **_):
    return AdMSoftmaxLoss(in_features, out_features, s=30.0, m=0.4)

def ls_cross_entropy(**__):
    return LabelSmoothingCrossEntropy()

def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)


if __name__ == "__main__":
    config = edict()
    config.loss = edict()
    config.loss.name = 'arcface'
    config.loss.params = edict()
    # config.loss.params.in_features = 512
    # config.loss.params.out_features = 1108

    criterion = get_loss(config)

    # input_ = torch.randn((8, 512))
    # label_ = torch.tensor([1, 0, 0, 0] * 2).unsqueeze(1)

    # print(label_)
    # print(label_.size())

    # print(criterion(input_, label_))