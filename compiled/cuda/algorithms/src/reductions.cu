#include "../include/basic_operations.cuh"
#include <stdexcept>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>


template<typename T>
float fdsp::GetMean(const T *d_pArray, const std::vector<size_t>& dimSizes, int axis)
{
    if(axis > (int)dimSizes.size() - 1)
        throw std::runtime_error("Array of dimension " + std::to_string(dimSizes.size()) + " has no axis " + std::to_string(axis));

    float tmp;
    int size = 1;
    for(auto& sz: dimSizes)
        size *= sz;

    if(axis == -1)
    {
        tmp = thrust::reduce(thrust::device, d_pArray, d_pArray + size, 0.0, thrust::plus<float>());
        tmp /= size;
    }
    float out = static_cast<float>(tmp);

    return out;
}

template float fdsp::GetMean<unsigned char>(const unsigned char *d_pArray, const std::vector<size_t> &dimSizes, int axis);
template float fdsp::GetMean<int>(const int *d_pArray, const std::vector<size_t> &dimSizes, int axis);
template float fdsp::GetMean<float>(const float *d_pArray, const std::vector<size_t> &dimSizes, int axis);
template float fdsp::GetMean<double>(const double *d_pArray, const std::vector<size_t> &dimSizes, int axis);


template<typename T>
struct addComplex
{
    __host__ __device__
    T operator()(const T& c1, const T& c2)
    {
        T out;
        out.x = c1.x + c2.x;
        out.y = c1.y + c2.y;
        return out;
    }
};

template<typename T>
cuComplex fdsp::GetMeanComplex(const T* d_pArray, const std::vector<size_t>& dimSizes, int axis)
{
    T tmp;
    int size = 1;
    for(auto& sz: dimSizes)
        size *= sz;

    T start = {0, 0};
    tmp = thrust::reduce(thrust::device, d_pArray, d_pArray + size, start, addComplex<T>());
    tmp.x /= size;
    tmp.y /= size;

    cuComplex out;
    out.x = static_cast<float>(tmp.x);
    out.y = static_cast<float>(tmp.y);

    return out;
}

template cuComplex fdsp::GetMeanComplex<cuComplex>(const cuComplex* d_pArray, const std::vector<size_t>& dimSizes, int axis);
template cuComplex fdsp::GetMeanComplex<cuDoubleComplex>(const cuDoubleComplex* d_pArray, const std::vector<size_t>& dimSizes, int axis);
