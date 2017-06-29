# pytorch-caffe-models
common caffe models in PyTorch format.

## how to use it

You need to have an environment with both Caffe and PyTorch installed.
Only tested with Python 3.6. Under root directory of project, run the following commands.

~~~bash
# download original caffe model files
python download_all_models.py
# convert them to HDF5
python convert_all_models.py
# optional, just for debugging.
python load_all_models_and_debug.py
~~~
Once you finish the above steps, you don't need Caffe any more.