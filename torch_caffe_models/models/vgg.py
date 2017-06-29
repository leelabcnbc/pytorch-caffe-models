"""this file gives the information needed to reconstruct official Caffe models in PyTorch"""
import torch.nn as nn
from . import _register_dict
from torchvision.models import vgg16, vgg19

# first, the 227x227 AlexNet



_register_dict('VGG16', {
    'type': 'linear',
    'model': 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel',
    'prototxt': 'https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt',
    'id': 'vgg16',
    'torch_class': vgg16,
    'input_blob': 'data',  # it's likely that we have some multi input net. But that's really rare.
})

_register_dict('VGG19', {
    'type': 'linear',
    'model': 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel',
    'prototxt': 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f43eeefc869d646b449aa6ce66f87bf987a1c9b5/VGG_ILSVRC_19_layers_deploy.prototxt',
    'id': 'vgg19',
    'torch_class': vgg19,
    'input_blob': 'data',  # it's likely that we have some multi input net. But that's really rare.
})
