import numpy as np

from FastDSP.core import fsdp_context
from FastDSP.core import decide_type
from FastDSP.structures import _data_structures
from FastDSP.utils import _math_utils


class GPUArray:
    """
    Class that interfaces python arrays with
    GPU buffers.

    Attributes
    ---------------------
    array:
        array allocated in a device
    device: int
        device number where array is allocated
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
        """
        :param array: numpy array of type float32, float64, complex64 or complex128.
        :param device: Device id where to transfer array.
        """

        if fsdp_context['num_devices'] == 0:
            raise MemoryError("Device not found. Not possible to allocate array")

        if not isinstance(array, np.ndarray):
            raise TypeError("Only numpy arrays of types float32, float64, complex64 or complex128 are accepted")

        self.array = array
        self.ndim = self.array.ndim
        self.dtype = array.dtype
        self.size = self.array.size

        if array.dtype == np.float32:
            self.array = _data_structures.GPUArrayFloat(self.array)
        elif array.dtype == np.float64:
            self.array = _data_structures.GPUArrayDouble(self.array)
        elif array.dtype == np.complex64:
            self.array = _data_structures.GPUArrayComplexFloat(self.array)
        elif array.dtype == np.complex128:
            self.array = _data_structures.GPUArrayComplexDouble(self.array)
        else:
            raise ValueError("Only float32, float64, complex64,and complex128 arrays are supported")

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

    def __add__(self, other):
        """
        Addition between to arrays of the same shape.
        Both arrays have to be on the same device.

        :param other: GPUArray instance
        :return: GPUArray with the sum of self and other
        :raises: Assertion error if arrays are not on the same device or have different shapes or type.


        >>> array1 = np.random.randn(100, 100)
        >>> gpu_array1 = GPUArray(array1)
        >>> gpu_array2 = GPUArray(array1)
        >>> gpu_array3 = gpu_array1 + gpu_array2
        """
        assert(self.device == other.device)
        assert(self.shape == other.shape)
        assert(self.array.dtype == other.array.dtype) #TODO: Figure out an efficient way to do the casting operations

        dtype = decide_type(self.array.dtype, other.array.dtype)
        out = GPUArray(np.zeros((1, 1), dtype=dtype))
        out.array = _math_utils.add_gpu_arrays(self.array, other.array)

        out.dtype = dtype
        out.size = self.size
        out.device = self.device
        out.shape = self.shape

        return out

    def __sub__(self, other):
        """
        Subtraction between to arrays of the same shape.
        Both arrays have to be on the same device.

        :param other: GPUArray instance
        :return: GPUArray with the difference of self and other
        :raises: Assertion error if arrays are not on the same device or have different shapes or type.


        >>> array1 = np.random.randn(100, 100)
        >>> gpu_array1 = GPUArray(array1)
        >>> gpu_array2 = GPUArray(array1)
        >>> gpu_array3 = gpu_array1 - gpu_array2
        """
        assert(self.device == other.device)
        assert(self.shape == other.shape)
        assert(self.array.dtype == other.array.dtype) #TODO: Figure out an efficient way to do the casting operations

        dtype = decide_type(self.array.dtype, other.array.dtype)
        out = GPUArray(np.zeros((1, 1), dtype=dtype))
        out.array = _math_utils.subtract_gpu_arrays(self.array, other.array)

        out.dtype = dtype
        out.size = self.size
        out.device = self.device
        out.shape = self.shape

        return out

    def __mul__(self, other):
        """
        Pointwise multiplication between to arrays of the same shape.
        Both arrays have to be on the same device.

        :param other: GPUArray instance
        :return: GPUArray with the pointwise multiplication of self and other
        :raises: Assertion error if arrays are not on the same device or have different shapes or type.


        >>> array1 = np.random.randn(100, 100)
        >>> gpu_array1 = GPUArray(array1)
        >>> gpu_array2 = GPUArray(array1)
        >>> gpu_array3 = gpu_array1*gpu_array2
        """
        assert(self.device == other.device)
        assert(self.shape == other.shape)
        assert(self.array.dtype == other.array.dtype) #TODO: Figure out an efficient way to do the casting operations

        dtype = decide_type(self.array.dtype, other.array.dtype)
        out = GPUArray(np.zeros((1, 1), dtype=dtype))
        out.array = _math_utils.multiply_gpu_arrays(self.array, other.array)

        out.dtype = dtype
        out.size = self.size
        out.device = self.device
        out.shape = self.shape

        return out

    def __truediv__(self, other):
        """
        Pointwise division between to arrays of the same shape.
        Both arrays have to be on the same device.

        :param other: GPUArray instance
        :return: GPUArray with the pointwise division of self and other
        :raises: Assertion error if arrays are not on the same device or have different shapes or type.


        >>> array1 = np.random.randn(100, 100)
        >>> gpu_array1 = GPUArray(array1)
        >>> gpu_array2 = GPUArray(array1)
        >>> gpu_array3 = gpu_array1/gpu_array2
        """
        assert(self.device == other.device)
        assert(self.shape == other.shape)
        assert(self.array.dtype == other.array.dtype) #TODO: Figure out an efficient way to do the casting operations

        dtype = decide_type(self.array.dtype, other.array.dtype)
        out = GPUArray(np.zeros((1, 1), dtype=dtype))
        out.array = _math_utils.divide_gpu_arrays(self.array, other.array)

        out.dtype = dtype
        out.size = self.size
        out.device = self.device
        out.shape = self.shape

        return out

    def get(self):
        """
        :return: host numpy array with the values of GPUArray.

        >>> cpu_array = gpu_array.get()
        """
        return self.array.get().reshape(*self.shape)

