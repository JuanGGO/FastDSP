#include "../include/data_structures.cuh"
#include "../core/include/debugging.cuh"
#include <cuComplex.h>

#pragma region GPUArray

template<typename T>
fdsp::GPUArray<T>::GPUArray(const T* ptr, const std::vector<size_t>& dimSizes):
        m_dimSizes(dimSizes), ndims(dimSizes.size())
{
    m_size = 1;
    for(auto& size: m_dimSizes)
        m_size *= size;

    m_dArray.resize(m_size);
    CHECK(cudaMemcpy(GetPointerToArray(), ptr, sizeof(T)*m_size, cudaMemcpyHostToDevice));
}

template<typename T>
fdsp::GPUArray<T>::GPUArray(const std::vector<size_t> &dimSizes) :
    m_dimSizes(dimSizes), ndims(dimSizes.size())
{
    m_size = 1;
    for(auto& size: m_dimSizes)
        m_size *= size;

    m_dArray.resize(m_size);
    CHECK(cudaMemset(GetPointerToArray(), 0, sizeof(T)*m_size));
}

template<typename T>
fdsp::GPUArray<T>::GPUArray(const GPUArray<T>& array)
{
    m_dimSizes = array.m_dimSizes;
    ndims = array.ndims;
    m_size = array.m_size;
    m_dArray.resize(m_size);
    CHECK(cudaMemcpy(GetPointerToArray(), thrust::raw_pointer_cast(&array.m_dArray[0]), sizeof(T)*m_size, cudaMemcpyDeviceToDevice));
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

template<typename T>
std::vector<size_t> fdsp::GPUArray<T>::GetDimensionSizes() const
{
    return m_dimSizes;
}

template<typename T>
const T* fdsp::GPUArray<T>::GetPointerToArrayConst()const
{
    return thrust::raw_pointer_cast(&(m_dArray[0]));
}

template<typename T>
T* fdsp::GPUArray<T>::GetPointerToArray()
{
    return thrust::raw_pointer_cast(&(m_dArray[0]));
}

template<typename T>
size_t fdsp::GPUArray<T>::GetSize() const
{
    return m_size;
}

template<typename T>
thrust::device_vector<T> fdsp::GPUArray<T>::GetDeviceVector() const
{
    return m_dArray;
}

template class fdsp::GPUArray<unsigned char>;
template class fdsp::GPUArray<int>;
template class fdsp::GPUArray<float>;
template class fdsp::GPUArray<double>;
template class fdsp::GPUArray<cuComplex>;
template class fdsp::GPUArray<cuDoubleComplex>;
#pragma endregion







