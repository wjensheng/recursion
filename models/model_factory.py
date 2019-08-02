import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchvision

from .pooling import MAC, SPoC, GeM, RMAC, Rpool
from .normalization import L2N, PowerLaw
from datasets import get_dataset, get_dataloader

# possible global pooling layers, each on of these can be made regional
POOLING = {
    'mac': MAC,
    'spoc': SPoC,
    'gem': GeM,
    'rmac': RMAC,
}

# output dimensionality for supported architectures
OUTPUT_DIM = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'densenet121': 1024,
    'densenet161': 2208,
    'densenet169': 1664,
    'densenet201': 1920,
    'squeezenet1_0': 512,
    'squeezenet1_1': 512,
}

class RecursionNet(nn.Module):

    def __init__(self, features, lwhiten, pool, whiten, meta):
        super(RecursionNet, self).__init__()

        # transfer weight from pretrained network
        trained_kernel = features[0].weight

        new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)

        features[0] = new_conv

        self.features = nn.Sequential(*features)
        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta

    def forward(self, x):
        # x -> features
        o = self.features(x)

        # TODO: properly test (with pre-l2norm and/or post-l2norm)
        # if lwhiten exist: features -> local whiten
        if self.lwhiten is not None:
            # o = self.norm(o)
            s = o.size()
            o = o.permute(0, 2, 3, 1).contiguous().view(-1, s[1])
            o = self.lwhiten(o)
            o = o.view(s[0], s[2], s[3], self.lwhiten.out_features).permute(0, 3, 1, 2)
            # o = self.norm(o)

        # features -> pool -> norm
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o))

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1, 0)

    def __repr__(self):
        tmpstr = super(RecursionNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n'  # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     local_whitening: {}\n'.format(self.meta['local_whitening'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     regional: {}\n'.format(self.meta['regional'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr


def init_network(params):
    # parse params with default values
    architecture = params.get('architecture', 'resnet18')
    local_whitening = params.get('local_whitening', False)
    pooling = params.get('pooling', 'gem')
    regional = params.get('regional', False)
    whitening = params.get('whitening', False)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)

    # get output dimensionality size
    dim = OUTPUT_DIM[architecture]

    # loading network from torchvision
    if pretrained:        
        # initialize with network pretrained on imagenet in pytorch
        net_in = getattr(torchvision.models, architecture)(pretrained=True)
    else:
        # initialize with random weights
        net_in = getattr(torchvision.models, architecture)(pretrained=False)

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]

    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))

    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())

    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))
    
    # initialize local whitening
    # if local_whitening:
    lwhiten = None

    # initialize pooling
    pool = POOLING[pooling]()

    # initialize regional pooling
    if regional:
        rpool = pool
        rwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: rwhiten with possible dimensionality reduce

        if pretrained:
            rw = '{}-{}-r'.format(architecture, pooling)
            print(">> {}: for '{}' there is no regional whitening computed, random weights are used"
                    .format(os.path.basename(__file__), rw))

        pool = Rpool(rpool, rwhiten)

    # initialize whitening
    # if whitening:
    whiten = None

    # create meta information to be stored in the network
    meta = {
        'architecture': architecture,
        'local_whitening': local_whitening,
        'pooling': pooling,
        'regional': regional,
        'whitening': whitening,
        'mean': mean,
        'std': std,
        'outputdim': dim,
    }

    # create a generic image retrieval network
    net = RecursionNet(features, lwhiten, pool, whiten, meta)

    return net 


def extract_vectors(net, loader, ms=[1], msp=1, print_freq=10):
    # moving network to gpu and eval mode
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # extracting vectors
    with torch.no_grad():
        vecs = torch.zeros(net.meta['outputdim'], len(loader.dataset))
        for i, input in enumerate(loader):
            if torch.cuda.is_available(): input = input.cuda()

            if len(ms) == 1:
                vecs[:, i] = extract_ss(net, input)
            else:
                vecs[:, i] = extract_ms(net, input, ms, msp)

            if (i + 1) % print_freq == 0 or (i + 1) == len(loader.dataset):
                print('\r>>>> {}/{} done...'.format((i + 1), 
                                                    len(loader.dataset)), 
                                                    end='')
        print('')

    return vecs


def extract_ss(net, input):
    return net(input).cpu().data.squeeze()


def extract_ms(net, input, ms, msp):
    v = torch.zeros(net.meta['outputdim'])

    for s in ms:
        if s == 1:
            input_t = input.clone()
        else:
            input_t = nn.functional.interpolate(input, 
                                                scale_factor=s, 
                                                mode='bilinear', 
                                                align_corners=False)

        v += net(input_t).pow(msp).cpu().data.squeeze()

    v /= len(ms)
    v = v.pow(1. / msp)
    v /= v.norm()

    return v    


def extract_regional_vectors(net, loader, ms=[1], msp=1, print_freq=10):
    # moving network to gpu and eval mode
    if torch.cuda.is_available(): net.cuda()
    net.eval()

    # extracting vectors
    with torch.no_grad():
        vecs = []
        for i, input in enumerate(loader):
            if torch.cuda.is_available(): input = input.cuda()

            if len(ms) == 1:
                vecs.append(extract_ssr(net, input))
            else:
                # TODO: not implemented yet
                # vecs.append(extract_msr(net, input, ms, msp))
                raise NotImplementedError

            if (i + 1) % print_freq == 0 or (i + 1) == len(loader.dataset):
                print('\r>>>> {}/{} done...'.format((i + 1), 
                                                    len(loader.dataset)), 
                                                    end='')
        print('')

    return vecs


def extract_ssr(net, input):
    return (net.pool(net.features(input), aggregate=False).squeeze(0)
                                                          .squeeze(-1)
                                                          .squeeze(-1)
                                                          .permute(1, 0)
                                                          .cpu().data)



def extract_local_vectors(net, loader, ms=[1], msp=1, print_freq=10):
    # moving network to gpu and eval mode
    if torch.cuda.is_available(): net.cuda()
    net.eval()

    # extracting vectors
    with torch.no_grad():
        vecs = []
        for i, input in enumerate(loader):
            if torch.cuda.is_available(): input = input.cuda()

            if len(ms) == 1:
                vecs.append(extract_ssl(net, input))
            else:
                # TODO: not implemented yet
                # vecs.append(extract_msl(net, input, ms, msp))
                raise NotImplementedError

            if (i + 1) % print_freq == 0 or (i + 1) == len(loader.dataset):
                print('\r>>>> {}/{} done...'.format((i + 1), len(loader.dataset)), end='')
        print('')

    return vecs


def extract_ssl(net, input):
    return (net.norm(net.features(input))
                        .squeeze(0)
                        .view(net.meta['outputdim'], -1)
                        .cpu().data)



def test():
    model_params = {}
    model_params['architecture'] = 'resne18'
    model_params['pooling'] = 'gem'
    model_params['local_whitening'] = False
    model_params['regional'] = False
    model_params['whitening'] = False
    # model_params['mean'] = ...  # will use default
    # model_params['std'] = ...  # will use default
    model_params['pretrained'] = True
    model = init_network(model_params)
    print(model)


if __name__ == "__main__":
    pass    