#include <cufft.h>
#include "../structures/include/data_structures.cuh"


namespace fdsp
{
#pragma region Device API
    GPUArray<cuComplex> FourierTransform(const GPUArray<float>& array, int axis);
#pragma endregion
}