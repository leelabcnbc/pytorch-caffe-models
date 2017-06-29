model_def_dict = {}


def _register_dict(name, properties):
    model_def_dict[name] = properties

# load all modules to register them.
# `import *` doesn't work. Need to do this explicitly.
from . import caffe_official, vgg
