from libcpp.vector cimport vector


ctypedef unsigned char uint8


#TODO: make the path relative to what the client has
cdef extern from "/usr/local/cuda-10.2/include/vector_types.h" :
    struct float2:
        float x
        float y

    struct double2:
        double x
        double y


cdef extern from "../../../compiled/cuda/structures/include/data_structures.cuh" namespace "fdsp":
    cdef cppclass GPUArray[T]:
        GPUArray(const T* ptr, const vector[size_t]& dimSizes)
        GPUArray(const vector[size_t]& dimSizes)
        GPUArray(const GPUArray[T]& array)
        void Get(T *h_ptr)const
        T GetElement(size_t index)const
        T* GetPointerToArray()
        const T* GetPointerToArrayConst()const


