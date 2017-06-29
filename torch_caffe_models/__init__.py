import os.path

cache_dir = os.path.normpath(os.path.join(os.path.split(__file__)[0], 'cache'))

assert os.path.isabs(cache_dir)
