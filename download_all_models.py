"""script to predownload all the caffe models
since this is not an industry-grade stuff, I just make this as simple as possible: you need to download first.

the downloading will be extremely simple: use curl
"""

# this is to load all the models

import os
from subprocess import run

from torch_caffe_models import cache_dir
from torch_caffe_models.models import model_def_dict

for model_name, model_properties in model_def_dict.items():
    print(model_name)
    # put them under `cache_dir/id/`,
    # as `model.caffemodel`
    dir_to_save = os.path.join(cache_dir, model_properties['id'])
    os.makedirs(dir_to_save, exist_ok=True)
    # download. will overwrite old ones (well this is not that bad for now).
    model_file = os.path.join(dir_to_save, 'model.caffemodel')
    prototxt_file = os.path.join(dir_to_save, 'model.prototxt')
    if not os.path.exists(model_file):
        run(['curl', '-o', model_file, model_properties['model']], check=True)
    if not os.path.exists(prototxt_file):
        run(['curl', '-o', prototxt_file, model_properties['prototxt']], check=True)
