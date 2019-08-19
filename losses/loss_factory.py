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

        loss = self.classify_loss(output, labels)

        # loss1 = self.classify_loss(output, labels)
        # loss2 = self.classify_loss(cosine, labels)
        # gamma=1
        # loss=(loss1+gamma*loss2)/(1+gamma)

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

        loss = self.classify_loss(output, labels)

        # loss1 = self.classify_loss(output, labels)
        # loss2 = self.classify_loss(cosine, labels)
        # gamma=1
        # loss=(loss1+gamma*loss2)/(1+gamma)

        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 's=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class AdaCosLoss(nn.modules.Module):
    
    def __init__(self, in_features, out_features, m=0.50, ls_eps=0, theta_zero=math.pi/4):
        super(AdaCosLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.in_features = in_features
        self.out_features = out_features
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
            B_avg = torch.sum(B_avg) / logits.size(0)
            theta_med = torch.median(theta)
            self.s = torch.log(B_avg) / torch.cos(torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med))
        output *= self.s
        # return output

        loss = self.classify_loss(output, labels)

        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'm=' + str(self.m) \
               + ', s=' + str(self.s) \
               + ', in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'


class SphereFaceLoss(nn.Module):

    def __init__(self, s=30.0, m=1.35):
        super(SphereFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.m = m
        
    def forward(self, logits, labels):
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s
        
        loss = self.classify_loss(output, labels)
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 's=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class AdMSoftmaxLoss(nn.Module):
    """https://github.com/cvqluu/Additive-Margin-Softmax-Loss-Pytorch"""

    def __init__(self, s=30.0, m=0.4):
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        
    def forward(self, logits, labels):        
        # assert len(x) == len(labels)
        # assert torch.min(labels) >= 0
        # assert torch.max(labels) < self.out_features
        
        wf = logits
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 's=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


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


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class NormSoftmaxLoss(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """
    def __init__(self, temperature=0.05):
        super(NormSoftmaxLoss, self).__init__()
                
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, instance_targets):        
        prediction_logits = logits
        loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)
        return loss

    def __repr__(self):
        return self.__class__.__name__ + '(' \
                + 'temperature=' + str(self.temperature) + ')'


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        #dist = []
        #for i in range(batch_size):
        #    value = distmat[i][mask[i]]
        #    value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #    dist.append(value)
        #dist = torch.cat(dist)
        #loss = dist.mean()
        return loss


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)

    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)

    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)

        dist_mat = euclidean_dist(global_feat, global_feat)

        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
            
        return loss, dist_ap, dist_an


def cross_entropy(**_):
    return torch.nn.CrossEntropyLoss()

def focal(**_):
    return FocalLoss(gamma=1)

def arcface(**_):
    return ArcFaceLoss(s=65.0, m=0.5, easy_margin=False)    

def cosface(**_):
    return CosFaceLoss(s=30.0, m=0.40) 

def adacos(in_features, out_features, **_):
    return AdaCosLoss(in_features, out_features, m=0.50, ls_eps=0, theta_zero=math.pi/4)

def amsoftmax(in_features, out_features, **_):
    return AdMSoftmaxLoss(s=30.0, m=0.4)

def normsoftmax(in_features, out_features, **_):
    return NormSoftmaxLoss(temperature=0.05)

def sphereface(**_):
    return SphereFaceLoss(s=30.0, m=1.35)

def ls_cross_entropy(**__):
    return LabelSmoothingCrossEntropy()

def center_loss(num_classes=1108, feat_dim=2048, use_gpu=True, **_):
    return CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=use_gpu)

def triplet_loss(margin=None, **_):
    return TripletLoss(margin=margin)    

def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)


if __name__ == '__main__':
    use_gpu = False
    center_loss = center_loss(use_gpu=use_gpu)
    features = torch.rand(16, 2048)
    targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).long()
    if use_gpu:
        features = torch.rand(16, 2048).cuda()
        targets = torch.Tensor([0, 1, 2, 3, 2, 3, 1, 4, 5, 3, 2, 1, 0, 0, 5, 4]).cuda()

    loss = center_loss(features, targets)
    print(loss)