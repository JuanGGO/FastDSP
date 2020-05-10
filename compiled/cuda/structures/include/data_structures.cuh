#pragma once
#include<vector>
#include "../../core/include/cuda_helper.h"
#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <complex.h>


namespace fdsp
{

    template<typename T>
    class GPUArray
    {
    private:
        std::vector<size_t> m_dimSizes;
        size_t m_size;
        thrust::device_vector<T> m_dArray;

    public:
        int ndims;

        GPUArray(const T* ptr, const std::vector<size_t>& dimSizes);
        GPUArray(const std::vector<size_t>& dimSize);
        GPUArray(const GPUArray<T>& array);

        void Get(T* h_ptr)const;
        T GetElement(size_t index)const;
        std::vector<size_t> GetDimensionSizes()const;
        const T* GetPointerToArrayConst()const;
        T* GetPointerToArray();
        size_t GetSize()const;
        thrust::device_vector<T> GetDeviceVector()const;
    };

}

