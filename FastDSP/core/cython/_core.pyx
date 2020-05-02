

cdef extern from "../../../compiled/cuda/core/include/initialization.cuh" namespace "fdsp":
    int GetDeviceCount()


cpdef int get_device_count():
    return GetDeviceCount()
