import torch
import numpy as np

class Resize_preprocess(object):
    """Rescales the input PIL.Image to the given 'size_w,size_h'.
    """

    def __init__(self, size_w,size_h):
        self.size = (size_w,size_h)

    def __call__(self, img):
        return img.resize(self.size)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def calculate_mean_std(loader):
    # the image should be preprocessed by torch.transform.ToTensor(), so the value is in [0,1]
    sum=np.ones(3)
    cnt=0
    for datas,_ in loader:
        cnt+=len(datas)
        for data in datas:
            data=data.numpy()
            sum+=data.sum(1).sum(1)/np.prod(data.shape[1:])
    mean=sum/cnt
    error=np.ones(3)
    _mean=mean.reshape([3,1,1])
    for datas,_ in loader:
        cnt+=len(datas)
        for data in datas:
            data=data.numpy()
            error+=((data-_mean)**2).sum(1).sum(1)/np.prod(data.shape[1:])
    std=np.sqrt(error/cnt)
    return mean,std

def no_strict_load_state_dict(net, state_dict):
    r"""Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True``, then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :meth:`~torch.nn.Module.state_dict` function.

    Arguments:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        strict (bool, optional): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    """
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(net)