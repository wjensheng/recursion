import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

import models.pooling as pooling
from models.metric_learning import ArcMarginProduct, AddMarginProduct, AdaCos

class RcicNet(nn.Module):

    DIVIDABLE_BY = 32

    def __init__(self,
                 n_classes,
                 model_name='resnet50',
                 pool='GeM',
                 args_pooling: dict={},
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(RcicNet, self).__init__()        

        self.backbone = getattr(pretrainedmodels, model_name)(num_classes=1000)

        # transfer weight from pretrained network
        trained_kernel = self.backbone.conv1.weight

        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)

        self.backbone.conv1 = new_conv

        final_in_features = self.backbone.last_linear.in_features

        # HACK: work around for this issue https://github.com/Cadene/pretrained-models.pytorch/issues/120
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.pooling = getattr(pooling, pool)(**args_pooling)

        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_feat(x)
        if self.loss_module in ('arcface', 'cosface', 'adacos'):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x

def get_model(config):
    n_classes = config.model.num_classes
    model_name = config.model.arch
    pool = config.model.pool
    args_pooling = {}
    use_fc = config.model.use_fc
    fc_dim = config.model.fc_dim
    dropout = config.model.dropout
    loss_module = config.model.loss_module
    s = config.model.self
    margin = config.model.margin    

    net = RcicNet(n_classes, model_name, pool, args_pooling,
                  use_fc, fc_dim, dropout, loss_module, 
                  s, margin)
    return net

if __name__ == "__main__":
    net = RcicNet(1108, 'resnet18')

    t = torch.randn(16, 6, 512, 512)
    labels = torch.randn(16,)

    # print(labels.size())

    print(net(t, labels).size())