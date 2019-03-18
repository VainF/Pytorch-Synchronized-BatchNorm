import torch.nn as nn
from .syncbn import SyncBatchNorm

def convert_sync_batchnorm(module):
    module_output = module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module_output = SyncBatchNorm(module.num_features,
                                      module.eps, module.momentum,
                                      module.affine,
                                      module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, convert_sync_batchnorm(child))
    return module_output