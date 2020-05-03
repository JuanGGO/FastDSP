import numpy as np

from FastDSP.core import fsdp_context
from FastDSP.structures import _data_structures


class GPUArray:
    """
    Class that interfaces python arrays with
    GPU buffers.

    Attributes
    ---------------------
    array: array
        A list, tuple or numpy array.
    device: int
        Device id where to transfer array.
    size: int
        array size.
    shape: tuple
        Shape of the array (as in numpy.shape).
    ndim: int
        Number of dimensions of the array

    Examples
    -------------------------------------------

    >>> import numpy as np
    >>> array = np.ones((5, 4), dtype=np.float32)
    >>> gpu_array = GPUArray(array)
    """

    def __init__(self, array, device=0):

        if fsdp_context['num_devices'] == 0:
            raise MemoryError("Device not found. Not possible to allocate array")

        if isinstance(array, list) or isinstance(array, tuple):
            self.array = np.asarray(array, dtype=np.float32)
        if isinstance(array, np.ndarray):
            self.array = array

        self.ndim = self.array.ndim
        dtype = self.array.dtype
        self.size = self.array.size

        if array.dtype == np.float32:
            self.array = _data_structures.ArrayGPUFloat(self.array)
        elif array.dtype == np.float64:
            self.array = _data_structures.ArrayGPUDouble(self.array)
        else:
            raise ValueError("Only float32 and float64 arrays are supported")

        self.device = device
        self.shape = array.shape

    def __getitem__(self, item):
        """
        Returns the item indexed by item.
        Throws ValueError if item >= self.size.

        :param item: index of element to return.
        :return: array element at item.
        :raises: ValueError if item >= self.size

        >>> value = gpu_array[item]
        """
        if item >= self.size:
            raise ValueError("index {} out of range".format(item))
        return self.array[item]

    def get(self):
        """
        :return: Host GPUArray instance.

        >>> cpu_array = gpu_array.get()
        """
        return self.array.get()



