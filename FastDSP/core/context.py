"""
Contains all methods necessary to create the context about
the machine where FastDSP is running. e.g. number of gpus,
number of cores, etc. The information retrieved is represented
in a dictionary called fdsp_context.
"""


from . import _core

fsdp_context = {
    'num_devices': _core.get_device_count()
}


