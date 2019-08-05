import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from easydict import EasyDict as edict

class ToyNet(nn.Module):

    def __init__(self, n_classes, model_name='resnet50', use_fc=False, fc_dim=512, dropout=0.0, loss_module='softmax'):
        super(ToyNet, self).__init__()        

        self.backbone = getattr(pretrainedmodels, model_name)(num_classes=1000)

        trained_kernel = self.backbone.conv1.weight

        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)

        self.backbone.conv1 = new_conv

        final_in_features = self.backbone.last_linear.in_features

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_feat(x)
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
    use_fc = config.model.use_fc
    fc_dim = config.model.fc_dim
    dropout = config.model.dropout
    loss_module = config.model.loss_module
    
    net = ToyNet(n_classes, model_name, use_fc, fc_dim, dropout, loss_module)

    return net

if __name__ == "__main__":
    cfg = edict()
    cfg.model = edict()
    cfg.model.arch = 'resnet18'
    cfg.model.dropout = 0
    cfg.model.loss_module = 'softmax' 
    cfg.model.use_fc = False
    cfg.model.fc_dim = 512
    cfg.model.image_size = 224 # resize
    cfg.model.num_classes = 1200
    cfg.model.pretrained = True
    cfg.model.lr = 3e-4

    model = get_model(cfg)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1: 
        model = torch.nn.DataParallel(model)

    if torch.cuda.is_available(): 
        model = model.cuda()

    input_ = torch.randn((8, 6, 224, 224))
    label_ = torch.randn((8, 6))
    print(model(input_, label_))