#pragma once
#include <cuComplex.h>

namespace fdsp
{
    void PrintDevicePtr(const float* d_ptr, int N);
    void PrintDevicePtr(const double *d_ptr, int N);
    void PrintDevicePtr(const cuComplex* d_ptr, int N);
}