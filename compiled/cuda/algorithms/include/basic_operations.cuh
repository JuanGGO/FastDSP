#pragma once
#include <cuda_runtime_api.h>
#include <vector>
#include <cuComplex.h>

#pragma region reductions
namespace fdsp
{
    template<typename T>
    float GetMean(const T *d_pArray, const std::vector<size_t>& dimSizes, int axis=-1);

    template<typename T>
    cuComplex GetMeanComplex(const T* d_pArray, const std::vector<size_t>& dimSizes, int axis=-1);
}
#pragma endregion