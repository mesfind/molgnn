
from types import ModuleType
from importlib import import_module


import torch_geometric.loader
import torch_geometric.transforms
import torch_geometric.utils
import torch_geometric.profile

from .seed import seed_everything
from .home import get_home_dir, set_home_dir
import molgnn.data
from .pygdata import RedDB
from .molgnn import MolGCNConv, MolGATConv
from .molfeatures import GenMolGraph, GenMolecules, GenMolFeatures
from .utils/utils.py import NoamLR


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
# python/util/lazy_loader.py
class LazyLoader(ModuleType):
    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super().__init__(name)

    def _load(self):
        module = import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)


datasets = LazyLoader('data', globals(), 'molgnn.data')
graphgym = LazyLoader('graphgym', globals(), 'molgnn.graphgym')

__version__ = '1.0'

__all__ = [
    'seed_everything',
    'get_home_dir',
    'set_home_dir',
    'RedDB',
    'MolGCNConv',
    'MolGATConv',
    'GenMolGraph',
    'GenMolecules',
    'GenMolFeatures',
    'torch_geometric',
    '__version__',
]
