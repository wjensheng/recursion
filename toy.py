import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

class RecursionNet(nn.Module):

    def __init__(self, n_classes=1108, model_name='resnet18', 
                 fc_dim=512, loss_module='softmax'):
        super(RecursionNet, self).__init__()        
                
        self.backbone = getattr(pretrainedmodels, model_name)(num_classes=1000)
        self.loss_module = loss_module

        final_in_features = self.backbone.last_linear.in_features
        
        if 'resnet' in model_name:
            trained_kernel = self.backbone.conv1.weight

            new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

            with torch.no_grad():
                new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)

            self.backbone.conv1 = new_conv
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.pooling = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        x = self.backbone(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        return x


if __name__ == "__main__":
    model = RecursionNet()

    input_ = torch.randn((16, 6, 224, 224))
    label_ = torch.tensor([1, 2, 3, 4] * 4)

    output = model(input_)

    print(output.size())