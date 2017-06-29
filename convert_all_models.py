"""script to predownload all the caffe models
since this is not an industry-grade stuff, I just make this as simple as possible: you need to download first.

the downloading will be extremely simple: use curl
"""

# this is to load all the models

import os
from subprocess import run

from torch_caffe_models.models import model_def_dict
from torch_caffe_models.models import converter

for model_name, model_properties in model_def_dict.items():
    print(model_name)
    if model_properties['type'] == 'linear':
        converter.convert_linear_model(model_properties)
    else:
        raise ValueError('unsupported type')

    # # then load model
    # model_cls = model_properties['torch_class']()
    # for x, y in model_cls.named_parameters():
    #     print(x, y.size())
