"""
Contains all methods necessary to create the context about
the machine where FastDSP is running. e.g. number of gpus,
number of cores, etc. The information retrieved is represented
in a dictionary called fdsp_context.
"""


from . import _core
from tools.load_config import get_cuda_include_and_lib_folders

cuda_folders_location = get_cuda_include_and_lib_folders()

if cuda_folders_location is None:
    fdsp_context = {
        'has_cuda': False,
        'num_devices': 0
    }
else:
    fdsp_context = {
        'has_cuda': True,
        'cuda_include_folder': cuda_folders_location['include'],
        'cuda_libs_folder': cuda_folders_location['lib64'],
        'num_devices': _core.get_device_count()
    }

