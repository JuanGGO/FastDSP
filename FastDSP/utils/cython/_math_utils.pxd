from _data_structures cimport BaseGPUArray


cdef add_gpu_arrays(BaseGPUArray arr1,
                    BaseGPUArray arr2)

cpdef subtract_gpu_arrays(BaseGPUArray arr1,
                          BaseGPUArray arr2)

cpdef multiply_gpu_arrays(BaseGPUArray arr1,
                          BaseGPUArray arr2)

cpdef divide_gpu_arrays(BaseGPUArray arr1,
                        BaseGPUArray arr2)
