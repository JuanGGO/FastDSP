import numpy as np

cimport _cuda_algorithms
from _data_structures cimport BaseGPUArray
from _cuda_structures cimport uint8
from _cuda_structures cimport float2 as cuComplex
from _cuda_structures cimport double2 as cuDoubleComplex
from FastDSP.core import decide_type


cpdef float get_mean(BaseGPUArray arr, int axis=-1):

    dtype = decide_type(arr.dtype, arr.dtype)

    if dtype == np.uint8:
        return _cuda_algorithms.GetMean[uint8](arr._uint8thisptr[0].GetPointerToArrayConst(), arr.dim_sizes, axis)
    elif dtype == np.int32:
        return _cuda_algorithms.GetMean[int](arr._ithisptr[0].GetPointerToArrayConst(), arr.dim_sizes, axis)
    elif dtype == np.float32:
        return _cuda_algorithms.GetMean[float](arr._fthisptr[0].GetPointerToArrayConst(), arr.dim_sizes, axis)
    elif dtype == np.float64:
        return _cuda_algorithms.GetMean[double](arr._dthisptr[0].GetPointerToArrayConst(), arr.dim_sizes, axis)


cpdef complex get_mean_complex(BaseGPUArray arr, int axis=-1):

    cdef:
        complex out;
        cuComplex _out;

    dtype = decide_type(arr.dtype, arr.dtype)

    if dtype == np.complex64:
        _out =  _cuda_algorithms.GetMeanComplex[cuComplex](arr._cfthisptr[0].GetPointerToArrayConst(), arr.dim_sizes, axis)
    elif dtype == np.complex128:
        _out = _cuda_algorithms.GetMeanComplex[cuDoubleComplex](arr._cdthisptr[0].GetPointerToArrayConst(), arr.dim_sizes, axis)

    out.real = _out.x; out.imag = _out.y

    return out


