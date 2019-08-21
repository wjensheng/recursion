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
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x
            

class RecursionNet(nn.Module):

    def __init__(self, num_classes, model_name='resnet50', 
                 fc_dim=512, loss_module='softmax'):
        super(RecursionNet, self).__init__()        
                
        # self.backbone = getattr(pretrainedmodels, model_name)(num_classes=1000)

        # final_in_features = self.backbone.last_linear.in_features        

        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=False)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

        trained_kernel = self.model.conv1.weight            
        self.model.conv1 = create_new_conv(trained_kernel)

                            
        # self.pooling = AdaptiveConcatPool2d()
        # self.flatten = Flatten()        
        # self.bn1 = nn.BatchNorm1d(1024 * self.expand)
        # self.dropout1 = nn.Dropout(p=0.25)
        # self.fc1 = nn.Linear(1024 * self.expand, 512 * self.expand)
        # self.relu = nn.ReLU(inplace=True)
        # self.bn2 = nn.BatchNorm1d(512 * self.expand)   
        # self.dropout2 = nn.Dropout(p=0.5)     
        # self._init_params()        
    
        # final_in_features = fc_dim * self.expand
        
        # if loss_module == 'arcface':
        #     self.final = ArcMarginProduct(final_in_features, n_classes)
        # elif loss_module == 'cosface':
        #     self.final = AddMarginProduct(final_in_features, n_classes)
        # elif loss_module == 'adacos':
        #     self.final = AdaCos(final_in_features, n_classes)
        # elif loss_module == 'sphereface':
        #     self.final = SphereProduct(final_in_features, n_classes)
        # elif loss_module == 'amsoftmax':
        #     self.final = AdaptiveMargin(final_in_features, n_classes)
        # else:
        #     self.final = nn.Linear(final_in_features, n_classes)

    # def _init_params(self):
    #     nn.init.kaiming_normal_(self.fc1.weight)
    #     nn.init.constant_(self.fc1.bias, 0)
    #     nn.init.constant_(self.bn1.weight, 1)
    #     nn.init.constant_(self.bn1.bias, 0)
    #     nn.init.constant_(self.bn2.weight, 1)
    #     nn.init.constant_(self.bn2.bias, 0)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return torch.mean(torch.stack(y), dim=0)

def get_model(config):
    num_classes = config.model.num_classes
    model_name = config.model.arch
    fc_dim = config.model.fc_dim
    loss_module = config.loss.name

    net = RecursionNet(num_classes=num_classes, model_name=model_name,
                       fc_dim=fc_dim, loss_module=loss_module)
                  
    return net