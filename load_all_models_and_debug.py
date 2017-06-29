"""script to predownload all the caffe models
since this is not an industry-grade stuff, I just make this as simple as possible: you need to download first.

the downloading will be extremely simple: use curl
"""

# this is to load all the models

import os
from subprocess import run

from torch_caffe_models.models import model_def_dict
from torch_caffe_models.models import loader

for model_name, model_properties in model_def_dict.items():
    print(model_name)
    if model_properties['type'] == 'linear':
        loader.load_linear_model(model_properties, debug=True)
    else:
        raise ValueError('unsupported type')
