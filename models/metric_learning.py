# https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
# https://github.com/4uiiurz1/pytorch-adacos/blob/master/metrics.py
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.nn import Parameter
import math


class AdaCos(nn.Module):
    def __init__(self, in_features, out_features, m=0.50, ls_eps=0, theta_zero=math.pi/4):
        super(AdaCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        return logits
        # # add margin
        # theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        # target_logits = torch.cos(theta + self.m)
        # one_hot = torch.zeros_like(logits)
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # if self.ls_eps > 0:
        #     one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # output = logits * (1 - one_hot) + target_logits * one_hot
        # # feature re-scale
        # with torch.no_grad():
        #     B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
        #     B_avg = torch.sum(B_avg) / input.size(0)
        #     theta_med = torch.median(theta)
        #     self.s = torch.log(B_avg) / torch.cos(torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med))
        # output *= self.s

        # return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features        
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # self.easy_margin = easy_margin
        # self.cos_m = math.cos(m)
        # self.sin_m = math.sin(m)
        # self.th = math.cos(math.pi - m)
        # self.mm = math.sin(math.pi - m) * m

    def forward(self, input):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine

        # sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # phi = cosine * self.cos_m - sine * self.sin_m
        # if self.easy_margin:
        #     phi = torch.where(cosine > 0, phi, cosine)
        # else:
        #     phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # # --------------------------- convert label to one-hot ---------------------------
        # # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')

        # if torch.cuda.is_available():
        #     one_hot = torch.zeros(cosine.size(), device='cuda')
        # else:
        #     one_hot = torch.zeros(cosine.size())

        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # if self.ls_eps > 0:
        #     one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # output *= self.s

        # return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine

        # phi = cosine - self.m
        # # --------------------------- convert label to one-hot ---------------------------
        # if torch.cuda.is_available():
        #     one_hot = torch.zeros(cosine.size(), device='cuda')
        # else:
        #     one_hot = torch.zeros(cosine.size())

        # # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        # output *= self.s
        # # print(output)

        # return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'


class SphereProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=1.35):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        return logits

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class AdaptiveMargin(nn.Module):
    def __init__(self, in_features, out_features):
        super(AdaptiveMargin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        logits = self.fc(x)

        return logits

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'


class EmbeddedFeatureWrapper(nn.Module):
    """
    Wraps a base model with embedding layer modifications.
    """
    def __init__(self, in_features, out_features):
        super(EmbeddedFeatureWrapper, self).__init__()

        self.standardize = nn.LayerNorm(in_features, elementwise_affine=False)

        self.remap = None
        if in_features != out_features:
            self.remap = nn.Linear(in_features, out_features, bias=False)

        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)


    def forward(self, input):
        x = self.standardize(input)

        if self.remap:
            x = self.remap(x)

        x = F.normalize(x, dim=1)
        
        norm_weight = nn.functional.normalize(self.weight, dim=1)

        logits = nn.functional.linear(x, norm_weight)

        return logits


    def __str__(self):
        return self.__class__.__name__ + '(' \
                + 'in_features=' + str(self.in_features) \
                + ', out_features=' + str(self.out_features) + ')'