from sys import path
import numpy as np

from FastDSP.algorithms.cython import _reductions


def mean(array, axis=-1):
    '''
    Calculates the mean of a gpu array on a given axis

    :param array: Instance of a class with class memeber array
    :param axis: axis to use to take the mean
    :return: mean value of the array around axis

    >>> import FastDSP as fdsp
    >>> import numpy as np
    >>> gpu_array = fdsp.GPUArray(np.random.randn(4, 4))
    >>> print(mean(gpu_array))
    '''

    if array.dtype == np.complex64 or array.dtype == np.complex128:
        out = _reductions.get_mean_complex(array.array, axis)
    else:
        out = _reductions.get_mean(array.array, axis)

    return out

