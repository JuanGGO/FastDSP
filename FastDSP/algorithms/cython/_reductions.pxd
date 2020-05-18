import numpy as np

cimport _cuda_algorithms
from _data_structures cimport BaseGPUArray
from _cuda_structures cimport uint8
from FastDSP.core import decide_type


cpdef float get_mean(const BaseGPUArray arr, int axis=-1)

cpdef complex get_mean_complex(BaseGPUArray arr, int axis=-1)
