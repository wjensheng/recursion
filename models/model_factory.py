import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels
import easydict as edict

import models.pooling as pooling
from models.metric_learning import ArcMarginProduct, AddMarginProduct, AdaCos, SphereProduct

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class RcicNet(nn.Module):

    DIVIDABLE_BY = 32

    def __init__(self,
                 n_classes,
                 model_name='resnet50',
                 pool='GeM',
                 args_pooling: dict={},
                 use_fc=True,
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
        final_in_features = self.backbone.last_linear.in_features        
        
        # transfer weight from pretrained network
        trained_kernel = self.backbone.conv1.weight            

        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)

        self.backbone.conv1 = new_conv

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
                
        self.pooling = AdaptiveConcatPool2d() # self.pooling = getattr(pooling, pool)(**args_pooling)

        self.use_fc = use_fc
        if use_fc:
            self.bn1 = nn.BatchNorm1d(1024)
            self.fc1 = nn.Linear(1024, 512)
            self.bn2 = nn.BatchNorm1d(512)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(512, 512)
            self.bn3 = nn.BatchNorm1d(512)
            self._init_params()

            # self.bn = nn.BatchNorm1d(fc_dim)
            # self.dropout = nn.Dropout(p=dropout)
            # self.fc = nn.Linear(final_in_features, fc_dim)
            # self.relu = nn.ReLU(inplace=True)                        
            # self._init_params()

            final_in_features = fc_dim

        else:
            self.fc1 = nn.Linear(1024, 512)

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        elif loss_module == 'sphere': # TODO: fix wrapper
            self.final = SphereProduct(final_in_features, n_classes)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)
        nn.init.constant_(self.bn3.weight, 1)
        nn.init.constant_(self.bn3.bias, 0)

    def forward(self, x):
        feature = self.extract_feat(x)
        if self.loss_module in ('arcface', 'cosface', 'adacos', 'sphere'):
            logits = self.final(feature)
        else:
            logits = self.final(feature)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)

        # x = self.pooling(x).view(batch_size, -1)
        x = self.pooling(x)

        x = x.view(x.size(0), -1)

        # if self.use_fc:
        x = self.bn1(x)
        x = F.dropout(x, p=0.25)
        x = self.fc1(x)
        x = self.relu(x)        
        x = self.bn2(x)
        x = F.dropout(x, p=0.5)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        x = self.bn3(x)   

        # else:
        #     x = self.fc1(x)
        
        #     x = self.bn(x)
        #     x = self.dropout(x)
        #     x = self.fc(x)
        #     x = self.relu(x)
        #     x = self.bn(x)
        #     x = self.dropout(x)

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
    s = config.model.s
    margin = config.model.margin

    net = RcicNet(n_classes, model_name, pool, args_pooling,
                  use_fc, fc_dim, dropout, loss_module, 
                  s, margin)
    return net

if __name__ == "__main__":
    cfg = edict()
    cfg.model = edict()
    cfg.model.arch = 'resnet18'
    cfg.model.pool = 'GeM' 
    cfg.model.local_whitening = False
    cfg.model.use_fc = False
    cfg.model.fc_dim = 512
    cfg.model.dropout = 0
    cfg.model.loss_module = 'arcface' # 'arcface', 'cosface', 'softmax'
    cfg.model.s = 30.0
    cfg.model.margin = 0.5
    cfg.model.regional = False
    cfg.model.whitening = False
    cfg.model.image_size = 512 # resize
    cfg.model.num_classes = 1108
    cfg.model.pretrained = True
    cfg.model.lr = 3e-4
    cfg.model.load_trained = True
    print(get_model(cfg))
