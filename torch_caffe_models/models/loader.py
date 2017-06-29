"""this module has all methods needed to convert a Caffe model to PyTorch model.
Different models may need different conversion processes.
"""

import os
# simplest one would be those linear models.
# simply for loop over PyTorch modules for those with parameters (conv and fc)
# and filling them with corresponding params from Caffe models would be fine.
import os.path
from collections import OrderedDict
from functools import partial

import h5py
import numpy as np
from numpy.linalg import norm
from torch import FloatTensor
from torch import nn
from torch.autograd import Variable

from .layers import SpatialCrossMapLRN
from .. import cache_dir


# this would make caffe and pytorch result more similar.
# anyway. I found that different networks in Caffe, different settings on cudnn will match it best.
# cudnn.enabled = XXX

def load_one_dataset(f_handle, dataset_to_load):
    data = f_handle[dataset_to_load][...]
    if 'name' in f_handle[dataset_to_load].attrs:
        name = f_handle[dataset_to_load].attrs['name']
    else:
        name = None
    return data, name


def _forward_hook(m, in_, out_, module_name, callback_dict):
    # if callback_dict[module_name]['type'] == LayerType.RELU:
    assert isinstance(out_, Variable)
    assert 'output' not in callback_dict[module_name], 'same module called twice!'
    # I use Tensor so that during backwards,
    # I don't have to think about moving numpy array to/from devices.
    callback_dict[module_name]['output'] = out_.data.clone()
    # print(module_name, callback_dict[module_name]['output'].size())


def _augment_module_pre(net: nn.Module, module_types=None) -> (dict, list):
    callback_dict = OrderedDict()  # not necessarily ordered, but this can help some readability.

    forward_hook_remove_func_list = []

    if module_types is None:
        # by default, only ReLU will count.
        module_types = (nn.ReLU,)
    assert len(module_types) > 0
    for x, y in net.named_modules():
        if any(isinstance(y, c) for c in module_types):
            callback_dict[x] = {}
            forward_hook_remove_func_list.append(
                y.register_forward_hook(partial(_forward_hook, module_name=x, callback_dict=callback_dict)))

    def remove_handles():
        for h in forward_hook_remove_func_list:
            h.remove()

    return callback_dict, remove_handles


def load_linear_model(model_properties, debug=False):
    # first, load files
    model_dir = os.path.join(cache_dir, model_properties['id'])
    model_hdf5 = os.path.join(model_dir, 'model.hdf5')

    model = model_properties['torch_class']()
    model.eval()

    with h5py.File(model_hdf5, 'r') as f_out:
        for param_idx, (param_name, param) in enumerate(model.named_parameters()):
            data, name = load_one_dataset(f_out, f'/weights/{param_idx}')
            print(f'{name} -> {param_name}')
            assert data.shape == tuple(param.data.size())
            # assign data
            param.data[...] = FloatTensor(data)

        if debug:
            callback_dict, remove_handles = _augment_module_pre(model,
                                                                (nn.ReLU,
                                                                 SpatialCrossMapLRN,
                                                                 nn.MaxPool2d))
            input_data = load_one_dataset(f_out, '/debug/input')[0]
            # somehow, LRN layer doesn't work correctly under CPU.
            # well. check test_LRN.py. looks like a numerical issue, instead of wrong implementation.
            model.cuda()
            model(Variable(FloatTensor(input_data).cuda()))
            # model(Variable(FloatTensor(input_data)))
            # then check the callback
            for blob_idx, (x, y) in enumerate(callback_dict.items()):
                ref_data, name = load_one_dataset(f_out, f'/debug/{blob_idx}')
                print(f'{name} -> {x}')
                data_this = y['output'].cpu().numpy()
                assert data_this.shape == ref_data.shape
                assert np.all(np.isfinite(data_this))
                assert np.all(np.isfinite(ref_data))
                norm_ratio = norm((data_this - ref_data).ravel()) / norm(ref_data.ravel())
                print(data_this.mean(), data_this.std(), norm_ratio)
                assert norm_ratio < 1e-5
            model.cpu()
    return model

# then test model's feedforward.
