from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .loss_factory import get_loss

def make_loss_with_center(config):
    if config.model.model_name == 'resnet18' or config.model.model_name == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if config.loss.name == 'center':
        center_criterion = CenterLoss(num_classes=config.model.num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif config.loss.name == 'triplet_center':
        triplet = TripletLoss(config.loss.params.margin)  # triplet loss
        center_criterion = CenterLoss(num_classes=config.model.num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    else:
        print('expected loss with center should be center, triplet_center'
              'but got {}'.format(config.loss.name))
        
    xent = CrossEntropyLabelSmooth(num_classes=config.model.num_classes)
    print("label smooth on, numclasses:", config.model.num_classes)


    def loss_func(score, feat, target):
        if config.loss.name == 'center':            
            return xent(score, target) + \
                   config.loss.params.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            
        elif config.loss.name == 'triplet_center':            
            return xent(score, target) + \
                   triplet(feat, target)[0] + \
                   config.loss.params.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
        
    return loss_func, center_criterion