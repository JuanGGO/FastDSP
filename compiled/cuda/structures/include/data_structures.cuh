#include<vector>
#include "../../core/include/cuda_helper.h"
#include <cuda_runtime_api.h>

namespace fdsp
{
    template<typename T>
    class GPUArray
    {
    private:
        std::vector<size_t> m_dimSizes;
        size_t m_size;
        T* m_dPtr;

    public:
        int dims;

        GPUArray(const T* ptr, const std::vector<size_t>& dimSizes);

        void Get(T* h_ptr)const;
        T Get(size_t index)const;
    };
}


template<typename T>
fdsp::GPUArray<T>::GPUArray(const T* ptr, const std::vector<size_t>& dimSizes):
        m_dimSizes(dimSizes), dims(dimSizes.size())
{
    m_size = 0;
    for(auto& size: m_dimSizes)
        m_size *= size;
    CHECK(cudaMalloc((void**)&m_dPtr, sizeof(T)*m_size));
    CHECK(cudaMemcpy(m_dPtr, ptr, sizeof(float)*m_size, cudaMemcpyHostToDevice));
}

template<typename T>
void fdsp::GPUArray<T>::Get(T *h_ptr) const
{
    CHECK(cudaMemcpy(h_ptr, m_dPtr, sizeof(float)*m_size, cudaMemcpyDeviceToHost));
}

template<typename T>
T fdsp::GPUArray<T>::Get(size_t index) const
{
    T* pOut = (T*)malloc(sizeof(T));
    T* aux;
    CHECK(cudaMalloc((void**)&aux, sizeof(T)*m_size));
    CHECK(cudaMemcpy(&aux, &m_dPtr, sizeof(T*), cudaMemcpyHostToHost));
    CHECK(cudaMemcpy(pOut, &aux[index], sizeof(T), cudaMemcpyDeviceToHost));
    T out = pOut[0];
    free(pOut);
    return out;
}
