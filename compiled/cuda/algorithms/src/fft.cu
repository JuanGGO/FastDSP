#include "../include/fft_related.cuh"
#include "../core/include/cuda_helper.h"
#include <cufft.h>


namespace fdsp
{

    GPUArray<cuComplex> FourierTransform(const GPUArray<float>& array, int axis)
    {
        int rank = array.ndims;
        int *n = (int*)malloc(sizeof(int)*rank);
        std::vector<size_t> temp = array.GetDimensionSizes();
        for(int i = 0; i < rank; i++)
            n[i] = (int)temp[i];
        int size = array.GetSize();
        GPUArray<cuComplex> out(array.GetDimensionSizes());

        cufftHandle plan;
        CHECK_CUFFT(cufftCreate(&plan));
        switch (axis)
        {
            case -1:
            {
                CHECK_CUFFT(cufftPlanMany(&plan, rank, (int*)n,
                                NULL, 1, size,
                                NULL, 1, size,
                                CUFFT_R2C, 1));
                break;
            }
            case 0:
                CHECK_CUFFT(cufftPlanMany(&plan, rank, (int*)n,
                                                (int*)n, n[0], 1, (int*)n, n[0],
                                                1, CUFFT_R2C, n[rank-1]));
                break;
            default:
                break;
        }

        CHECK_CUFFT(cufftExecR2C(plan, const_cast<float*>(array.GetPointerToArrayConst()), out.GetPointerToArray()));
        CHECK(cudaDeviceSynchronize());

        CHECK_CUFFT(cufftDestroy(plan));
        free(n);

        return out;
    }
}