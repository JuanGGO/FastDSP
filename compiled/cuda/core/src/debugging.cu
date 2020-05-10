#include "../include/debugging.cuh"
#include "../include/cuda_helper.h"
#include <cuda_runtime_api.h>
#include <cuComplex.h>


namespace fdsp
{
#pragma region kernels
    __global__ void
    KernelPrintDevicePtr(const float *d_ptr, int N)
    {
        for(int i = 0; i < N; i++)
        {
            printf("%f ", d_ptr[i]);
            if((i+1) % 20 == 0)
                printf("\n");
        }
    }

    __global__ void
    KernelPrintDevicePtr(const double *d_ptr, int N)
    {
        for(int i = 0; i < N; i++)
        {
            printf("%f ", d_ptr[i]);
            if((i+1) % 20 == 0)
                printf("\n");
        }
    }

    __global__ void
    KernelPrintDevicePtr(const cuComplex *d_ptr, int N)
    {
        for(int i = 0; i < N; i++)
        {
            printf("%f + i%f  ", d_ptr[i].x, d_ptr[i].y);
            if ((i + 1) % 10 == 0)
                printf("\n");
        }
    }

#pragma endregion

#pragma region Host API

    void PrintDevicePtr(const float* d_ptr, int N)
    {
        KernelPrintDevicePtr<<<1, 1>>>(d_ptr, N);
        CHECK(cudaDeviceSynchronize())
        CHECK(cudaGetLastError());
    }

    void PrintDevicePtr(const double *d_ptr, int N)
    {
        KernelPrintDevicePtr<<<1, 1>>>(d_ptr, N);
        CHECK(cudaDeviceSynchronize())
        CHECK(cudaGetLastError());
    }

    void PrintDevicePtr(const cuComplex* d_ptr, int N)
    {
        KernelPrintDevicePtr<<<1, 1>>>(d_ptr, N);
        CHECK(cudaDeviceSynchronize())
        CHECK(cudaGetLastError());
    }

#pragma endregion

}