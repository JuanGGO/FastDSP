from _cuda_structures cimport GPUArray

#region math declarations

cdef extern from "../../../compiled/cuda/utils/include/math.cuh" namespace "fdsp":
    void AddDevicePointers[T](const T* d_arr1, const T* d_arr2, int N, T* out)
    void SubtractDevicePointers[T](const T* d_arr1, const T* d_arr2, int N, T* out)
    void MultiplyDevicePointers[T](const T* d_arr1, const T* d_arr2, int N, T* out)
    void DivideDevicePointers[T](const T* d_arr1, const T* d_arr2, int N, T* out)


#endregion
