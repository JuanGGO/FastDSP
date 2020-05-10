from libcpp.vector cimport vector
cimport _cuda_structures as cuda
from _cuda_structures cimport float2 as cuComplex
from _cuda_structures cimport double2 as cuDoubleComplex


cdef class BaseGPUArray:
    cdef:
        size_t size
        int[::1] dims
        int ndims
        list dims_list
        vector[size_t] dim_sizes

        cuda.GPUArray[float] *_fthisptr
        cuda.GPUArray[double] *_dthisptr
        cuda.GPUArray[cuComplex] *_cfthisptr
        cuda.GPUArray[cuDoubleComplex] *_cdthisptr


cdef class GPUArrayFloat(BaseGPUArray):

    cpdef get(self)


cdef class GPUArrayDouble(BaseGPUArray):

    cpdef get(self)


cdef class GPUArrayComplexFloat(BaseGPUArray):
    cdef:
        cuComplex* v

    cpdef get(self)


cdef class GPUArrayComplexDouble(BaseGPUArray):
    cdef:
        cuDoubleComplex* v

    cpdef get(self)
