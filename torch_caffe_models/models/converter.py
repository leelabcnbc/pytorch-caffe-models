"""this module has all methods needed to convert a Caffe model to PyTorch model.
Different models may need different conversion processes.
"""

# simplest one would be those linear models.
# simply for loop over PyTorch modules for those with parameters (conv and fc)
# and filling them with corresponding params from Caffe models would be fine.
import os.path
import h5py
from .. import cache_dir
import os
import numpy as np

# suppress caffe warnings
# https://stackoverflow.com/questions/29788075/setting-glog-minloglevel-1-to-prevent-output-in-shell-from-caffe
os.environ['GLOG_minloglevel'] = '2'

import caffe

caffe.set_mode_gpu()


def save_one_dataset(f_handle, dataset_to_save, data, name=None):
    if dataset_to_save not in f_handle:
        f_handle.create_dataset(dataset_to_save, data=data)
        # follow <http://docs.h5py.org/en/latest/strings.html>
        if name is not None:
            f_handle[dataset_to_save].attrs['name'] = np.string_(name)
        f_handle.flush()
        print(f'{dataset_to_save} {name} done')
    else:
        print(f'{dataset_to_save} {name} done before')


def convert_linear_model(model_properties):
    # first, load files
    model_dir = os.path.join(cache_dir, model_properties['id'])
    model_caffemodel = os.path.join(model_dir, 'model.caffemodel')
    model_prototxt = os.path.join(model_dir, 'model.prototxt')
    model_hdf5 = os.path.join(model_dir, 'model.hdf5')

    # first load this model
    net = caffe.Net(model_prototxt,  # defines the structure of the model
                    model_caffemodel,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    # then first list all params
    param_idx = 0

    with h5py.File(model_hdf5) as f_out:
        for param_name, param in net.params.items():
            print(param_name, len(param))
            for param_idx_local, x in enumerate(param):
                param_name_this = f'{param_name}.{param_idx_local}'
                dataset_to_save = f'/weights/{param_idx}'
                save_one_dataset(f_out, dataset_to_save, x.data, param_name_this)
                param_idx += 1

        # ok. write test.
        # TODO: get input size from `data`. this is assumption.
        input_blob = model_properties['input_blob']
        input_size = net.blobs[input_blob].data.shape
        assert len(input_size) == 4
        input_size = (10,) + input_size[1:]
        # then reshape.
        net.blobs[input_blob].reshape(*input_size)
        # then feed data
        # then forward (you can't do backward, as there's no loss)
        # then collect all blobs, save them under `/test`
        rng_state = np.random.RandomState(seed=0)
        # 128 (well you can also use 127.5 I guess)  assumes that input originally lie in [0,255].
        input_to_use = rng_state.randn(*input_size) * 128
        input_to_use = input_to_use.astype(np.float32)
        net.blobs[input_blob].data[...] = input_to_use
        net.forward()

        blob_idx = 0
        for blob_name, blob in net.blobs.items():
            if blob_name == input_blob:
                continue

            blob_name_this = f'{blob_name}'
            dataset_to_save = f'/debug/{blob_idx}'
            save_one_dataset(f_out, dataset_to_save, blob.data, blob_name_this)
            blob_idx += 1

        # save input
        dataset_to_save = '/debug/input'
        save_one_dataset(f_out, dataset_to_save, input_to_use, input_blob)
