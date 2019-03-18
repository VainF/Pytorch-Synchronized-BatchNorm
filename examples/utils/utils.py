from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os 

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
def convert_bn2gn(module):
    mod = module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        num_features = module.num_features
        num_groups = num_features//16
        mod = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    for name, child in module.named_children():
        mod.add_module(name, convert_bn2gn(child))
    del module
    return mod

def group_params(net):
    group_decay = []
    group_no_decay = []
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):# or isinstance(m, nn.Linear):
            group_no_decay.extend(m.parameters(recurse=False))
        else:
            for name, params in m.named_parameters(recurse=False):
                if 'bias' in name:
                    group_no_decay.append(params)
                else:
                    group_decay.append(params)    
    return group_decay, group_no_decay
