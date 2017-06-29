"""script to predownload all the caffe models
since this is not an industry-grade stuff, I just make this as simple as possible: you need to download first.

the downloading will be extremely simple: use curl
"""

# this is to load all the models

from torch_caffe_models.models.layers import SpatialCrossMapLRN
import numpy as np
from torch.autograd import Variable
from torch import FloatTensor
from scipy.stats import pearsonr
from torch_caffe_models import cache_dir
import os.path
import h5py

if __name__ == '__main__':
    alexnet_ref = os.path.join(cache_dir, 'alexnet', 'model.hdf5')
    assert os.path.exists(alexnet_ref)
    with h5py.File(alexnet_ref, 'r') as f_out:
        data = f_out['/debug/0'][...]

    module = SpatialCrossMapLRN(size=5)
    # first, no cuda
    output1 = module(Variable(FloatTensor(data.copy()))).data.cpu().numpy()
    # then cuda
    module.cuda()
    output2 = module(Variable(FloatTensor(data.copy()).cuda())).data.cpu().numpy()
    print(output2.mean(), output2.std(), abs(output1 - output2).max())
    print(output1.mean(), output1.std(), abs(output1 - output2).max())
    print(pearsonr(output1.ravel(), output2.ravel())[0])
