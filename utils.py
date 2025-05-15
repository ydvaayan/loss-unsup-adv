import numpy as np
from PIL import Image


from torchvision.transforms import v2 as transforms

import torch
import torch.nn as nn
import sys
from tqdm import tqdm


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

random_transform = transforms.Compose([
    # transform,
    transforms.GaussianNoise(0., 0.05),
    # transforms.ElasticTransform(alpha=10., sigma=10.),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
])

normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


inverse_normalize = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
])

@torch.no_grad
def chunk_call(model, inputs, batchsize=128):
    if model is None:
        return torch.zeros(inputs.size(0), 4)
    outputs = []
    for i in tqdm(range(0, len(inputs), batchsize)):
        outputs.append(model(inputs[i:i+batchsize].cuda()).cpu())
    return torch.cat(outputs)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster