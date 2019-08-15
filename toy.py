import math
import torch
import torch.nn as nn
from torch.nn import Parameter

import pretrainedmodels
import torch.nn as nn

class NormNet(nn.Module):
    def __init__(self, input_dim=512, output_dim=2048):
        super(NormNet, self).__init__()
        self.backbone = pretrainedmodels.__dict__["resnet18"](num_classes=1000)

        trained_kernel = self.backbone.conv1.weight

        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)

        self.backbone.conv1 = new_conv

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.standardize = nn.LayerNorm(input_dim, elementwise_affine=False)

        self.remap = None
        if input_dim != output_dim:
            self.remap = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, images):
        x = self.backbone(images)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.standardize(x)

        if self.remap:
            x = self.remap(x)

        x = nn.functional.normalize(x, dim=1)

        return x


class NormSoftmaxLoss(nn.Module):
    """
    L2 normalize weights and apply temperature scaling on logits.
    """
    def __init__(self,
                 dim=2048,
                 num_instances=1108,
                 temperature=0.05):
        super(NormSoftmaxLoss, self).__init__()

        self.weight = Parameter(torch.Tensor(num_instances, dim))
        # Initialization from nn.Linear (https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/modules/linear.py#L129)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, embeddings, instance_targets):
        norm_weight = nn.functional.normalize(self.weight, dim=1)

        prediction_logits = nn.functional.linear(embeddings, norm_weight)

        loss = self.loss_fn(prediction_logits / self.temperature, instance_targets)
        return loss