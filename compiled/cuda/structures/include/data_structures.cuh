#include<vector>
#include "../../core/include/cuda_helper.h"
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>


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
        void Get(T* h_ptr)const;
        T GetElement(size_t index)const;
    };
}

