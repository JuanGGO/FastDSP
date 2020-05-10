import numpy as np
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
cimport _cuda_structures as cuda
from _cuda_structures cimport float2 as cuComplex
from _cuda_structures cimport double2 as cuDoubleComplex


cdef class BaseGPUArray:

    def __cinit__(self, array=None, list dimension_sizes=[]):
        if array is not None:
            self.dims_list = []
            self.ndims = array.ndim
            self.size = array.size
            self.dims = np.zeros((self.ndims, ), dtype=np.int32)
            self.dim_sizes.resize(self.ndims)
            for i in range(self.ndims):
                self.dims_list.append(array.shape[i])
                self.dim_sizes[i] = array.shape[i]
                self.dims[i] = array.shape[i]
        elif len(dimension_sizes) > 0:
            self.dims_list = dimension_sizes
            self.ndims = len(dimension_sizes)
            self.dims = np.zeros((self.ndims, ), dtype=np.int32)
            self.dim_sizes.resize(self.ndims)
            self.size = 1
            for i in range(self.ndims):
                self.dim_sizes[i] = dimension_sizes[i]
                self.dims[i] = dimension_sizes[i]
                self.size *= dimension_sizes[i]
        else:
            raise ValueError("Either the argument array or dimension_size should be given")


cdef class GPUArrayFloat(BaseGPUArray):

    def __cinit__(self, array=None, list dimension_sizes=[]):
        if array is None:
            array = np.zeros((self.size, ), dtype=np.float32)
        cdef:
            float[::1] v = np.ascontiguousarray(array.ravel())
        self._fthisptr = new cuda.GPUArray[float](&v[0], self.dim_sizes)

    def __dealloc__(self):
        if self._fthisptr:
            del self._fthisptr

    def __getitem__(self, index):
        return self._fthisptr[0].GetElement(<size_t>index)

    @property
    def dtype(self):
        return 'float'

    cpdef get(self):
        cdef:
            float[::1] out = np.zeros((self.size, ), dtype=np.float32)

        self._fthisptr[0].Get(&out[0])
        return np.asarray(out).reshape(*self.dims_list)


cdef class GPUArrayDouble(BaseGPUArray):

    def __cinit__(self, array=None, list dimension_sizes=[]):

        if array is None:
            array = np.zeros((self.size, ), dtype=np.float64)
        cdef:
            double[::1] v = np.ascontiguousarray(array.ravel())
        self._dthisptr = new cuda.GPUArray[double](&v[0], self.dim_sizes)

    def __dealloc__(self):
        if self._dthisptr:
            del self._dthisptr

    def __getitem__(self, index):
        return self._dthisptr.GetElement(<size_t>index)

    @property
    def dtype(self):
        return 'double'

    cpdef get(self):
        cdef:
            double[::1] out = np.zeros((self.size, ), dtype=np.float64)

        self._dthisptr[0].Get(&out[0])

        return np.asarray(out).reshape(*self.dims_list)

cdef class GPUArrayComplexFloat(BaseGPUArray):

    def __cinit__(self, array=None, list dimension_sizes=[]):
        cdef:
            int i
        self.v = <cuComplex*>malloc(sizeof(cuComplex)*self.size)
        if array is None:
            for i in range(self.size):
                self.v[i].x = 0
                self.v[i].y = 0
        else:
            array = array.ravel()
            for i in range(self.size):
                self.v[i].x = array[i].real
                self.v[i].y = array[i].imag

        self._cfthisptr = new cuda.GPUArray[cuComplex](self.v, self.dim_sizes)

    def __dealloc__(self):
        if self.v:
            free(self.v)
        if self._cfthisptr:
            del self._cfthisptr

    def __getitem__(self, index):
        value =  self._cfthisptr.GetElement(<int>index)
        return value.x + 1j*value.y

    @property
    def dtype(self):
        return 'complex_float'

    cpdef get(self):
        cdef:
            cuComplex* out = <cuComplex*>malloc(sizeof(cuComplex)*self.size)

        self._cfthisptr.Get(out)
        np_out = np.zeros((self.size, ), dtype=np.complex64)

        for i in range(self.size):
            np_out[i] = out[i].x + 1j*out[i].y

        free(out)

        return np_out.reshape(*self.dims_list)


cdef class GPUArrayComplexDouble(BaseGPUArray):

    def __cinit__(self, array=None, list dimension_sizes=[]):
        cdef:
            int i
        self.v = <cuDoubleComplex*>malloc(sizeof(cuDoubleComplex)*self.size)
        if array is None:
            for i in range(self.size):
                self.v[i].x = 0
                self.v[i].y = 0
        else:
            array = array.ravel()
            for i in range(self.size):
                self.v[i].x = array[i].real
                self.v[i].y = array[i].imag

        self._cdthisptr = new cuda.GPUArray[cuDoubleComplex](self.v, self.dim_sizes)

    def __dealloc__(self):
        if self.v:
            free(self.v)
        if self._cdthisptr:
            del self._thisptr

    def __getitem__(self, index):
        value =  self._cdthisptr.GetElement(<int>index)
        return value.x + 1j*value.y

    @property
    def dtype(self):
        return 'complex_double'

    cpdef get(self):
        cdef:
            cuDoubleComplex* out = <cuDoubleComplex*>malloc(sizeof(cuDoubleComplex)*self.size)

        self._cdthisptr.Get(out)
        np_out = np.zeros((self.size, ), dtype=np.complex128)

        for i in range(self.size):
            np_out[i] = out[i].x + 1j*out[i].y

        free(out)

        return np_out.reshape(*self.dims_list)
