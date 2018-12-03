'''
Many codes are borrowd from
https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
https://github.com/nakosung/VQ-VAE/blob/master/data_loader.py
'''
from torch.utils import data
from torchvision import datasets, transforms

import torch.utils.data.distributed as distributed

from .core.utils import rescaling
from .core import custom_datasets
from .core import custom_transforms

try:
    RESIZE = transforms.Resize
except AttributeError:
    RESIZE = transforms.Scale


def base_loader(dataset, **kwargs):
    shuffle = False if kwargs.get('sampler') else kwargs.get('shuffle', True)
    return data.DataLoader(dataset=dataset,
                           batch_size=kwargs.get('batch_size', 32),
                           shuffle=shuffle,
                           sampler=kwargs.get('sampler', None),
                           num_workers=kwargs.get('num_workers', 2),
                           pin_memory=kwargs.get('pin_memory', False))


def mnist(root, **kwargs):
    '''data_loader for MNIST
    root @str: path for dataset
    trasform @torchvision.transforms: custom transform
        (default: ToTensor(), rescaling)
    train @bool: return train or validation data_loader (default: True)
    download @bool: download data if not exists (default: False)
    batch_size @int: batch size (default: 32)
    shuffle @bool: shuffle (default: True)
    num_workers @int: num workers (default: 2)
    '''
    if kwargs.get('transform', None):
        transform = kwargs.get('transform')
    else:
        transform = transforms.Compose([transforms.ToTensor(), rescaling])
    dataset = datasets.MNIST(root,
                             train=kwargs.get('train', True),
                             download=kwargs.get('download', False),
                             transform=transform)
    return base_loader(dataset, **kwargs)


def __get_cifar_dataset(root, dataset, **kwargs):
    if kwargs.get('transform', None):
        transform = kwargs.get('transform')
    elif kwargs.get('is_classification', False):
        transform = custom_transforms.cifar_classification_transform(kwargs.get('random_crop', False))
    else:
        transform = transforms.Compose([transforms.ToTensor(), rescaling])
    return dataset(root,
                   train=kwargs.get('train', True),
                   download=kwargs.get('download', False),
                   transform=transform)


def cifar10(root, **kwargs):
    '''data_loader for CIFAR-10
    root @str: path for dataset
    trasform @torchvision.transforms: custom transform
        (default: ToTensor(), rescaling)
    train @bool: return train or validation data_loader (default: True)
    is_classification @bool: if True, use classification transform (default: False)
    random_crop @bool: if True, use random crop (default: False)
    download @bool: download data if not exists (default: False)
    batch_size @int: batch size (default: 32)
    shuffle @bool: shuffle (default: True)
    num_workers @int: num workers (default: 2)
    '''
    dataset = __get_cifar_dataset(root, datasets.CIFAR10, **kwargs)
    return base_loader(dataset, **kwargs)


def cifar100(root, **kwargs):
    '''data_loader for CIFAR-100
    root @str: path for dataset
    trasform @torchvision.transforms: custom transform
        (default: ToTensor(), rescaling)
    train @bool: return train or validation data_loader (default: True)
    is_classification @bool: if True, use classification transform (default: False)
    random_crop @bool: if True, use random crop (default: False)
    download @bool: download data if not exists (default: False)
    batch_size @int: batch size (default: 32)
    shuffle @bool: shuffle (default: True)
    num_workers @int: num workers (default: 2)
    '''
    dataset = __get_cifar_dataset(root, datasets.CIFAR100, **kwargs)
    return base_loader(dataset, **kwargs)


def ilsvr2015(root, **kwargs):
    '''data_loader for ILSVRC2015
    trasform @torchvision.transforms: custom transform
        (default: ToTensor(), rescaling)
    train @bool: return train or validation data_loader (default: True)
    is_classification @bool: if True, use classification transform (default: False)
    random_crop @bool: if True, use random crop (default: False)
    batch_size @int: batch size (default: 32)
    shuffle @bool: shuffle (default: True)
    distributed @bool: distributed (default: False)
    num_workers @int: num workers (default: 2)
    '''
    if kwargs.get('transform', None):
        transform = kwargs.get('transform')
    elif kwargs.get('is_classification', False):
        transform = custom_transforms.imagenet_classification_transform(kwargs.get('random_crop', False))
    else:
        transform = transforms.Compose([transforms.ToTensor(), rescaling])
    dataset = datasets.ImageFolder(root, transform)
    sampler = None
    if kwargs.get('train', True) and kwargs.get('distributed', False):
        sampler = distributed.DistributedSampler(dataset)
    return base_loader(dataset, sampler=sampler, **kwargs)


def tiny_imagenet200(root, **kwargs):
    '''data_loader for tiny-imagenet200
    '''
    raise NotImplemented


def celebA(root, **kwargs):
    '''data_loader for CelebA
    root @str: path for dataset
    trasform @torchvision.transforms: custom transform
        (default: CenterCrop(128), Resize(image_size), ToTensor(), rescaling)
    image_size @int: resize image size (default: None)
    batch_size @int: batch size (default: 32)
    shuffle @bool: shuffle (default: True)
    num_workers @int: num workers (default: 2)
    '''
    if kwargs.get('transform', None):
        transform = kwargs.get('transform')
    elif kwargs.get('image_size', None):
        transform = transforms.Compose([
            RESIZE(kwargs.get('image_size')),
            transforms.ToTensor(),
            rescaling])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            rescaling])
    dataset = custom_datasets.CelebImageFolder(root, transform)
    return base_loader(dataset, **kwargs)
