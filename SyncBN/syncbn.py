
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn as nn
import torch
import torch.nn.functional as F
from queue import Queue

from .functions import SyncBNFunction

import collections

_bn_context = collections.namedtuple("_bn_context", ['sync', 'is_master', 'cur_device', 'queue', 'devices'])

class SyncBatchNorm(_BatchNorm):
    """ Sync BN
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

        self.devices = list(range(torch.cuda.device_count()))
        self.sync = len(self.devices)>1
        self._slaves = self.devices[1:]
        self._queues = [ Queue(len(self._slaves)) ] + [ Queue(1) for _ in self._slaves ]

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError('expected at least 2 dims (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        
        if not self.training and self.track_running_stats:
            return F.batch_norm(input, running_mean=self.running_mean, running_var=self.running_var,
                         weight=self.weight, bias=self.bias, training=False, momentum=0.0, eps=self.eps)
        else:
            exponential_average_factor = 0.0
            if self.num_batches_tracked is not None: # track running statistics
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
            
            if input.is_cuda:
                cur_device = input.get_device()
                bn_ctx = _bn_context( self.sync, (cur_device==self.devices[0]), cur_device, self._queues, self.devices )
            else:
                bn_ctx = _bn_context( False, True, None, None, None  )
        
        return SyncBNFunction.apply( input, self.weight, self.bias, self.running_mean, self.running_var, exponential_average_factor, self.eps, self.training, bn_ctx )
    

class SyncBatchNorm1d(SyncBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


class SyncBatchNorm2d(SyncBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class SyncBatchNorm3d(SyncBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
