import numpy as np
from libc.stdio cimport printf

from _cuda_utils cimport AddDevicePointers, SubtractDevicePointers, MultiplyDevicePointers, DivideDevicePointers
from _data_structures cimport (GPUArrayFloat, GPUArrayDouble, GPUArrayComplexFloat, GPUArrayComplexDouble,
                                GPUArrayInt, GPUArrayUint8)
from _data_structures cimport BaseGPUArray
from _cuda_structures cimport GPUArray
from _cuda_structures cimport float2 as cuComplex
from _cuda_structures cimport double2 as cuDoubleComplex
from _cuda_structures cimport uint8
from FastDSP.core import decide_type


#region add_gpu_arrays

cpdef add_gpu_arrays(BaseGPUArray arr1,
                     BaseGPUArray arr2):
    cdef:
        int size = arr1.size
        GPUArray[uint8] *uint8out
        GPUArray[int] *iout
        GPUArray[float] *fout
        GPUArray[double] *dout
        GPUArray[cuComplex] *cfout
        GPUArray[cuDoubleComplex] *cdout

    dtype = decide_type(arr1.dtype, arr2.dtype)


    if dtype == np.uint8:
        uint8out = new GPUArray[uint8](arr1.dim_sizes)
        AddDevicePointers[uint8](arr1._uint8thisptr[0].GetPointerToArrayConst(),
                               arr2._uint8thisptr[0].GetPointerToArrayConst(),
                               size,
                               uint8out.GetPointerToArray())
        uint8_py_out = GPUArrayUint8(dimension_sizes=list(arr1.dims))
        uint8_py_out._uint8thisptr = uint8out
        return uint8_py_out
    elif dtype == np.int32:
        iout = new GPUArray[int](arr1.dim_sizes)
        AddDevicePointers[int](arr1._ithisptr[0].GetPointerToArrayConst(),
                                 arr2._ithisptr[0].GetPointerToArrayConst(),
                                 size,
                                 iout.GetPointerToArray())
        i_py_out = GPUArrayInt(dimension_sizes=list(arr1.dims))
        i_py_out._ithisptr = iout
        return i_py_out
    elif dtype == np.float32:
        fout = new GPUArray[float](arr1.dim_sizes)
        AddDevicePointers[float](arr1._fthisptr[0].GetPointerToArrayConst(),
                                 arr2._fthisptr[0].GetPointerToArrayConst(),
                                 size,
                                 fout.GetPointerToArray())
        f_py_out = GPUArrayFloat(dimension_sizes=list(arr1.dims))
        f_py_out._fthisptr = fout
        return f_py_out
    elif dtype == np.float64:
        dout = new GPUArray[double](arr1.dim_sizes)
        AddDevicePointers[double](arr1._dthisptr[0].GetPointerToArrayConst(),
                                 arr2._dthisptr[0].GetPointerToArrayConst(),
                                 size,
                                 dout.GetPointerToArray())
        d_py_out = GPUArrayDouble(dimension_sizes=list(arr1.dims))
        d_py_out._dthisptr = dout
        return d_py_out
    elif dtype == np.complex64:
        cfout = new GPUArray[cuComplex](arr1.dim_sizes)
        AddDevicePointers[cuComplex](arr1._cfthisptr[0].GetPointerToArrayConst(),
                                  arr2._cfthisptr[0].GetPointerToArrayConst(),
                                  size,
                                  cfout.GetPointerToArray())
        cf_py_out = GPUArrayComplexFloat(dimension_sizes=list(arr1.dims))
        cf_py_out._cfthisptr = cfout
        return cf_py_out
    elif dtype == np.complex128:
        cdout = new GPUArray[cuDoubleComplex](arr1.dim_sizes)
        AddDevicePointers[cuDoubleComplex](arr1._cdthisptr[0].GetPointerToArrayConst(),
                                     arr2._cdthisptr[0].GetPointerToArrayConst(),
                                     size,
                                     cdout.GetPointerToArray())
        cd_py_out = GPUArrayComplexDouble(dimension_sizes=list(arr1.dims))
        cd_py_out._cdthisptr = cdout
        return cd_py_out
    else:
        raise ValueError("Unknown type in add_gpu_arrays")

#endregion

#region subtract_gpu_arrays

cpdef subtract_gpu_arrays(BaseGPUArray arr1,
                          BaseGPUArray arr2):
    cdef:
        int size = arr1.size
        GPUArray[uint8] *uint8out
        GPUArray[int] *iout
        GPUArray[float] *fout
        GPUArray[double] *dout
        GPUArray[cuComplex] *cfout
        GPUArray[cuDoubleComplex] *cdout

    dtype = decide_type(arr1.dtype, arr2.dtype)

    if dtype == np.uint8:
        uint8out = new GPUArray[uint8](arr1.dim_sizes)
        SubtractDevicePointers[uint8](arr1._uint8thisptr[0].GetPointerToArrayConst(),
                                 arr2._uint8thisptr[0].GetPointerToArrayConst(),
                                 size,
                                 uint8out.GetPointerToArray())
        uint8_py_out = GPUArrayUint8(dimension_sizes=list(arr1.dims))
        uint8_py_out._uint8thisptr = uint8out
        return uint8_py_out
    elif dtype == np.int32:
        iout = new GPUArray[int](arr1.dim_sizes)
        SubtractDevicePointers[int](arr1._ithisptr[0].GetPointerToArrayConst(),
                               arr2._ithisptr[0].GetPointerToArrayConst(),
                               size,
                               iout.GetPointerToArray())
        i_py_out = GPUArrayInt(dimension_sizes=list(arr1.dims))
        i_py_out._ithisptr = iout
        return i_py_out
    elif dtype == np.float32:
        fout = new GPUArray[float](arr1.dim_sizes)
        SubtractDevicePointers[float](arr1._fthisptr[0].GetPointerToArrayConst(),
                                 arr2._fthisptr[0].GetPointerToArrayConst(),
                                 size,
                                 fout.GetPointerToArray())
        f_py_out = GPUArrayFloat(dimension_sizes=list(arr1.dims))
        f_py_out._fthisptr = fout
        return f_py_out
    elif dtype == np.float64:
        dout = new GPUArray[double](arr1.dim_sizes)
        SubtractDevicePointers[double](arr1._dthisptr[0].GetPointerToArrayConst(),
                                  arr2._dthisptr[0].GetPointerToArrayConst(),
                                  size,
                                  dout.GetPointerToArray())
        d_py_out = GPUArrayDouble(dimension_sizes=list(arr1.dims))
        d_py_out._dthisptr = dout
        return d_py_out
    elif dtype == np.complex64:
        cfout = new GPUArray[cuComplex](arr1.dim_sizes)
        SubtractDevicePointers[cuComplex](arr1._cfthisptr[0].GetPointerToArrayConst(),
                                     arr2._cfthisptr[0].GetPointerToArrayConst(),
                                     size,
                                     cfout.GetPointerToArray())
        cf_py_out = GPUArrayComplexFloat(dimension_sizes=list(arr1.dims))
        cf_py_out._cfthisptr = cfout
        return cf_py_out
    elif dtype == np.complex128:
        cdout = new GPUArray[cuDoubleComplex](arr1.dim_sizes)
        SubtractDevicePointers[cuDoubleComplex](arr1._cdthisptr[0].GetPointerToArrayConst(),
                                           arr2._cdthisptr[0].GetPointerToArrayConst(),
                                           size,
                                           cdout.GetPointerToArray())
        cd_py_out = GPUArrayComplexDouble(dimension_sizes=list(arr1.dims))
        cd_py_out._cdthisptr = cdout
        return cd_py_out
    else:
        raise ValueError("Unknown type in subtract_gpu_arrays")

#endregion

#region multiply_gpu_arrays

cpdef multiply_gpu_arrays(BaseGPUArray arr1,
                          BaseGPUArray arr2):
    cdef:
        int size = arr1.size
        GPUArray[uint8] *uint8out
        GPUArray[int] *iout
        GPUArray[float] *fout
        GPUArray[double] *dout
        GPUArray[cuComplex] *cfout
        GPUArray[cuDoubleComplex] *cdout

    dtype = decide_type(arr1.dtype, arr2.dtype)

    if dtype == np.uint8:
        uint8out = new GPUArray[uint8](arr1.dim_sizes)
        MultiplyDevicePointers[uint8](arr1._uint8thisptr[0].GetPointerToArrayConst(),
                                      arr2._uint8thisptr[0].GetPointerToArrayConst(),
                                      size,
                                      uint8out.GetPointerToArray())
        uint8_py_out = GPUArrayUint8(dimension_sizes=list(arr1.dims))
        uint8_py_out._uint8thisptr = uint8out
        return uint8_py_out
    elif dtype == np.int32:
        iout = new GPUArray[int](arr1.dim_sizes)
        MultiplyDevicePointers[int](arr1._ithisptr[0].GetPointerToArrayConst(),
                                    arr2._ithisptr[0].GetPointerToArrayConst(),
                                    size,
                                    iout.GetPointerToArray())
        i_py_out = GPUArrayInt(dimension_sizes=list(arr1.dims))
        i_py_out._ithisptr = iout
        return i_py_out
    if dtype == np.float32:
        fout = new GPUArray[float](arr1.dim_sizes)
        MultiplyDevicePointers[float](arr1._fthisptr[0].GetPointerToArrayConst(),
                                      arr2._fthisptr[0].GetPointerToArrayConst(),
                                      size,
                                      fout.GetPointerToArray())
        f_py_out = GPUArrayFloat(dimension_sizes=list(arr1.dims))
        f_py_out._fthisptr = fout
        return f_py_out
    elif dtype == np.float64:
        dout = new GPUArray[double](arr1.dim_sizes)
        MultiplyDevicePointers[double](arr1._dthisptr[0].GetPointerToArrayConst(),
                                       arr2._dthisptr[0].GetPointerToArrayConst(),
                                       size,
                                       dout.GetPointerToArray())
        d_py_out = GPUArrayDouble(dimension_sizes=list(arr1.dims))
        d_py_out._dthisptr = dout
        return d_py_out
    elif dtype == np.complex64:
        cfout = new GPUArray[cuComplex](arr1.dim_sizes)
        MultiplyDevicePointers[cuComplex](arr1._cfthisptr[0].GetPointerToArrayConst(),
                                          arr2._cfthisptr[0].GetPointerToArrayConst(),
                                          size,
                                          cfout.GetPointerToArray())
        cf_py_out = GPUArrayComplexFloat(dimension_sizes=list(arr1.dims))
        cf_py_out._cfthisptr = cfout
        return cf_py_out
    elif dtype == np.complex128:
        cdout = new GPUArray[cuDoubleComplex](arr1.dim_sizes)
        MultiplyDevicePointers[cuDoubleComplex](arr1._cdthisptr[0].GetPointerToArrayConst(),
                                                arr2._cdthisptr[0].GetPointerToArrayConst(),
                                                size,
                                                cdout.GetPointerToArray())
        cd_py_out = GPUArrayComplexDouble(dimension_sizes=list(arr1.dims))
        cd_py_out._cdthisptr = cdout
        return cd_py_out
    else:
        raise ValueError("Unknown type in multiply_gpu_arrays")

#endregion

#region divide_gpu_arrays

cpdef divide_gpu_arrays(BaseGPUArray arr1,
                          BaseGPUArray arr2):
    cdef:
        int size = arr1.size
        GPUArray[uint8] *uint8out
        GPUArray[int] *iout
        GPUArray[float] *fout
        GPUArray[double] *dout
        GPUArray[cuComplex] *cfout
        GPUArray[cuDoubleComplex] *cdout

    dtype = decide_type(arr1.dtype, arr2.dtype)

    if dtype == np.uint8:
        uint8out = new GPUArray[uint8](arr1.dim_sizes)
        DivideDevicePointers[uint8](arr1._uint8thisptr[0].GetPointerToArrayConst(),
                                      arr2._uint8thisptr[0].GetPointerToArrayConst(),
                                      size,
                                      uint8out.GetPointerToArray())
        uint8_py_out = GPUArrayUint8(dimension_sizes=list(arr1.dims))
        uint8_py_out._uint8thisptr = uint8out
        return uint8_py_out
    elif dtype == np.int32:
        iout = new GPUArray[int](arr1.dim_sizes)
        DivideDevicePointers[int](arr1._ithisptr[0].GetPointerToArrayConst(),
                                    arr2._ithisptr[0].GetPointerToArrayConst(),
                                    size,
                                    iout.GetPointerToArray())
        i_py_out = GPUArrayInt(dimension_sizes=list(arr1.dims))
        i_py_out._ithisptr = iout
        return i_py_out
    elif dtype == np.float32:
        fout = new GPUArray[float](arr1.dim_sizes)
        DivideDevicePointers[float](arr1._fthisptr[0].GetPointerToArrayConst(),
                                      arr2._fthisptr[0].GetPointerToArrayConst(),
                                      size,
                                      fout.GetPointerToArray())
        f_py_out = GPUArrayFloat(dimension_sizes=list(arr1.dims))
        f_py_out._fthisptr = fout
        return f_py_out
    elif dtype == np.float64:
        dout = new GPUArray[double](arr1.dim_sizes)
        DivideDevicePointers[double](arr1._dthisptr[0].GetPointerToArrayConst(),
                                       arr2._dthisptr[0].GetPointerToArrayConst(),
                                       size,
                                       dout.GetPointerToArray())
        d_py_out = GPUArrayDouble(dimension_sizes=list(arr1.dims))
        d_py_out._dthisptr = dout
        return d_py_out
    elif dtype == np.complex64:
        cfout = new GPUArray[cuComplex](arr1.dim_sizes)
        DivideDevicePointers[cuComplex](arr1._cfthisptr[0].GetPointerToArrayConst(),
                                          arr2._cfthisptr[0].GetPointerToArrayConst(),
                                          size,
                                          cfout.GetPointerToArray())
        cf_py_out = GPUArrayComplexFloat(dimension_sizes=list(arr1.dims))
        cf_py_out._cfthisptr = cfout
        return cf_py_out
    elif dtype == np.complex128:
        cdout = new GPUArray[cuDoubleComplex](arr1.dim_sizes)
        DivideDevicePointers[cuDoubleComplex](arr1._cdthisptr[0].GetPointerToArrayConst(),
                                                arr2._cdthisptr[0].GetPointerToArrayConst(),
                                                size,
                                                cdout.GetPointerToArray())
        cd_py_out = GPUArrayComplexDouble(dimension_sizes=list(arr1.dims))
        cd_py_out._cdthisptr = cdout
        return cd_py_out
    else:
        raise ValueError("Unknown type in divide_gpu_arrays")

#endregion
