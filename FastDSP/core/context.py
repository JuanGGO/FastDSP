from . import _core

fsdp_context ={
    'num_devices': _core.get_device_count()
}