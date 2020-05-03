from libcpp.vector cimport vector
import numpy as np

cdef extern from "../../../compiled/cuda/test_structures/include/data_structures.cuh" namespace "fdsp":
    cdef cppclass GPUArray[T]:
        GPUArray(const T* ptr, const vector[size_t]& dimSizes)
        void Get(T *h_ptr)const
        float GetElement(size_t index)const


cdef class ArrayGPUFloat:
    cdef:
        GPUArray[float] *_thisptr
        size_t size
        int[::1] dims
        int ndims
        list dims_list

    def __cinit__(self, array):
        cdef:
            vector[size_t] dim_sizes
            int d = array.ndim

        self.dims_list = []
        self.ndims = d
        self.size = array.size
        self.dims = np.zeros((self.ndims, ), dtype=np.int32)
        dim_sizes.resize(d)
        for i in range(d):
            self.dims_list.append(array.shape[i])
            dim_sizes[i] = array.shape[i]
            self.dims[i] = array.shape[i]

        cdef:
            float[::1] v = np.ascontiguousarray(array.ravel())
        self._thisptr = new GPUArray[float](&v[0], dim_sizes)

    def __dealloc__(self):
        if self._thisptr:
            del self._thisptr

    def __getitem__(self, index):
        return self._thisptr.GetElement(<size_t>index)

    cpdef get(self):
        cdef:
            float[::1] out = np.zeros((self.size, ), dtype=np.float32)

        self._thisptr.Get(&out[0])

        return np.asarray(out).reshape(*self.dims_list)


cdef class ArrayGPUDouble:
    cdef:
        GPUArray[double] *_thisptr
        size_t size
        int[::1] dims
        int ndims
        list dims_list

    def __cinit__(self, array):
        cdef:
            vector[size_t] dim_sizes
            int d = array.ndim

        self.dims_list = []
        self.ndims = d
        self.size = array.size
        self.dims = np.zeros((self.ndims, ), dtype=np.int32)
        dim_sizes.resize(d)
        for i in range(d):
            self.dims_list.append(array.shape[i])
            dim_sizes[i] = array.shape[i]
            self.dims[i] = array.shape[i]

        cdef:
            double[::1] v = np.ascontiguousarray(array.ravel())
        self._thisptr = new GPUArray[double](&v[0], dim_sizes)

    def __dealloc__(self):
        if self._thisptr:
            del self._thisptr

    def __getitem__(self, index):
        return self._thisptr.GetElement(<size_t>index)

    cpdef get(self):
        cdef:
            double[::1] out = np.zeros((self.size, ), dtype=np.float64)

        self._thisptr.Get(&out[0])

        return np.asarray(out).reshape(*self.dims_list)





