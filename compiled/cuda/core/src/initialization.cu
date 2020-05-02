#include "../include/initialization.cuh"
#include <cuda_runtime_api.h>
#include "../include/cuda_helper.h"

int fdsp::GetDeviceCount()
{
    int nDevices;
    CHECK(cudaGetDeviceCount(&nDevices));
    return nDevices;
}
