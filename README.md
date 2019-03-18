# SyncBatchNorm

Pytorch synchronized batch normalization implemented in pure python.

This repo is inspired by [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding).


# Requirements

pytorch >= 1.0

# Quick Start

It is recommended to convert your model to sync version using convert_sync_batchnorm. 

```python
from SyncBN import SyncBatchNorm2d
from SyncBN.utils import convert_sync_batchnorm
from torchvision.models import resnet34

sync_model = convert_sync_batchnorm( resnet34() ) # build resnet34 and replace bn with syncbn
sync_model = torch.nn.DataParallel(sync_model)    # Parallel on multi gpus
```

Or you can build your model from scratch.

```python
from SyncBN import SyncBatchNorm2d

sync_model = nn.Sequential(
                nn.Conv2d(3, 12, 3, 1, 1),
                SyncBatchNorm2d(12, momentum=0.1, eps=1e-5, affine=True),
                nn.ReLU(),
            )
sync_model = torch.nn.DataParallel(sync_model) # Parallel on multi gpus
```
# Cifar example

```bash
cd SyncBatchNorm/examples
python cifar.py --gpu_id 0,1 --data_root ./data --batch_size 64 --sync_bn
```

