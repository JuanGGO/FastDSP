import numpy as np

from FastDSP.core import fsdp_context
from FastDSP.structures import _data_structures


class GPUArray:

    def __init__(self, array, device=0):
        if fsdp_context['num_devices'] == 0:
            raise BufferError("Device not found. Not possible to allocate array")

        if isinstance(array, list) or isinstance(array, tuple):
            self.array = np.asarray(list)
        if isinstance(array, np.ndarray):
            self.array = array

        ndims = self.array.ndim
        dtype = self.array.dtype

        if array.dtype == np.float32:
            self.array = _data_structures.ArrayGPU_Float(self.array)
        else:
            raise ValueError("Only float32 arrays are supported")

    def __getitem__(self, item):
        return self.array[item]

    def get(self):
        return self.array.get()



