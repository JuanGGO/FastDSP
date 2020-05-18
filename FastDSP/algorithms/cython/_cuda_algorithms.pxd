from libcpp.vector cimport vector
from _cuda_structures cimport float2 as cuComplex


cdef extern from "../../../compiled/cuda/algorithms/include/basic_operations.cuh" namespace "fdsp":
    float GetMean[T](T *d_pArray, const vector[size_t]& dimSizes, int axis)
    cuComplex GetMeanComplex[T](const T *d_pArray, const vector[size_t]& dimSizes, int axis)
