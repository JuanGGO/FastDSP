#include "../include/data_structures.cuh"
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

template<typename T>
fdsp::GPUArray<T>::GPUArray(const T* ptr, const std::vector<size_t>& dimSizes):
        m_dimSizes(dimSizes), ndims(dimSizes.size())
{
    m_size = 1;
    for(auto& size: m_dimSizes)
        m_size *= size;

    m_dArray.resize(m_size);
    thrust::copy(ptr, ptr + m_size, m_dArray.begin());
}

template<typename T>
void fdsp::GPUArray<T>::Get(T *h_ptr) const
{
    thrust::copy(m_dArray.begin(), m_dArray.end(), h_ptr);
}

template<typename T>
T fdsp::GPUArray<T>::GetElement(size_t index) const
{
    return m_dArray[index];
}

template class fdsp::GPUArray<float>;
template class fdsp::GPUArray<double>;



