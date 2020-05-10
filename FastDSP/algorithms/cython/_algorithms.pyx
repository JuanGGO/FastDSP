from FastDSP.structures.cython._data_structures cimport GPUArrayFloat, GPUArrayDouble


cdef extern from "../../../compiled/cuda/algorithms/include/fft_related.cuh" namespace "fdsp":
    pass
