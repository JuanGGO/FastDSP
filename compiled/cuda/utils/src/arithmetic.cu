#include "../include/math.cuh"
#include<thrust/transform.h>
#include "../core/include/debugging.cuh"

#pragma region addition

struct Plus
{
    __device__
    float operator()(const float& x, const float& y)
    {
        return x + y;
    }
    __device__
    float operator()(const double& x, const double& y)
    {
        return x + y;
    }
    __device__
    cuComplex operator()(const cuComplex &x, const cuComplex &y)
    {
        cuComplex out;
        out.x = x.x + y.x;
        out.y = x.y + y.y;
        return out;
    }
    __device__
    cuDoubleComplex operator()(const cuDoubleComplex &x, const cuDoubleComplex &y)
    {
        cuDoubleComplex out;
        out.x = x.x + y.x;
        out.y = x.y + y.y;
        return out;
    }
};

template<typename T>
void fdsp::AddDevicePointers(const T *d_arr1, const T *d_arr2, int N, T *out)
{
        thrust::transform(thrust::device, d_arr1, d_arr1 + N, d_arr2, out, Plus());
}

template void fdsp::AddDevicePointers<float>(const float *d_arr1, const float *d_arr2, int N, float *out);
template void fdsp::AddDevicePointers<double>(const double *d_arr1, const double *d_arr2, int N, double *out);
template void fdsp::AddDevicePointers<cuComplex>(const cuComplex *d_arr1, const cuComplex *d_arr2, int N, cuComplex *out);
template void fdsp::AddDevicePointers<cuDoubleComplex>(const cuDoubleComplex *d_arr1, const cuDoubleComplex *d_arr2, int N, cuDoubleComplex *out);

template<typename T>
fdsp::GPUArray<T> fdsp::AddGPUArrays(const GPUArray<T>& arr1, const GPUArray<T>& arr2)
{
    GPUArray<T> out(arr1.GetDimensionSizes());
    fdsp::AddDevicePointers<T>(arr1.GetPointerToArrayConst(), arr2.GetPointerToArrayConst(),
                                arr1.GetSize(), out.GetPointerToArray());
    return out;
}

template fdsp::GPUArray<float> fdsp::AddGPUArrays<float>(const GPUArray<float>&, const GPUArray<float>&);
template fdsp::GPUArray<double> fdsp::AddGPUArrays<double>(const GPUArray<double>&, const GPUArray<double>&);
template fdsp::GPUArray<cuComplex> fdsp::AddGPUArrays<cuComplex>(const GPUArray<cuComplex>&, const GPUArray<cuComplex>&);
template fdsp::GPUArray<cuDoubleComplex> fdsp::AddGPUArrays<cuDoubleComplex>(const GPUArray<cuDoubleComplex>&, const GPUArray<cuDoubleComplex>&);

template<typename T>
void fdsp::AddGPUArrays(const GPUArray<T>& arr1, const GPUArray<T>& arr2, GPUArray<T>& out)
{
    fdsp::AddDevicePointers<T>(arr1.GetPointerToArrayConst(), arr2.GetPointerToArrayConst(),
                               arr1.GetSize(), out.GetPointerToArray());

}

template void fdsp::AddGPUArrays<float>(const GPUArray<float>&, const GPUArray<float>&, GPUArray<float>&);
template void fdsp::AddGPUArrays<double>(const GPUArray<double>&, const GPUArray<double>&, GPUArray<double>&);
template void fdsp::AddGPUArrays<cuComplex>(const GPUArray<cuComplex>&, const GPUArray<cuComplex>&, GPUArray<cuComplex>&);
template void fdsp::AddGPUArrays<cuDoubleComplex>(const GPUArray<cuDoubleComplex>&, const GPUArray<cuDoubleComplex>&, GPUArray<cuDoubleComplex>&);

#pragma endregion

#pragma region subtraction

struct Subtraction
{
    __device__
    float operator()(const float& x, const float& y)
    {
        return x - y;
    }
    __device__
    float operator()(const double& x, const double& y)
    {
        return x - y;
    }
    __device__
    cuComplex operator()(const cuComplex &x, const cuComplex &y)
    {
        cuComplex out;
        out.x = x.x - y.x;
        out.y = x.y - y.y;
        return out;
    }
    __device__
    cuDoubleComplex operator()(const cuDoubleComplex &x, const cuDoubleComplex &y)
    {
        cuDoubleComplex out;
        out.x = x.x - y.x;
        out.y = x.y - y.y;
        return out;
    }
};

template<typename T>
void fdsp::SubtractDevicePointers(const T *d_arr1, const T *d_arr2, int N, T *out)
{
    thrust::transform(thrust::device, d_arr1, d_arr1 + N, d_arr2, out, Subtraction());
}

template void fdsp::SubtractDevicePointers<float>(const float *d_arr1, const float *d_arr2, int N, float *out);
template void fdsp::SubtractDevicePointers<double>(const double *d_arr1, const double *d_arr2, int N, double *out);
template void fdsp::SubtractDevicePointers<cuComplex>(const cuComplex *d_arr1, const cuComplex *d_arr2, int N, cuComplex *out);
template void fdsp::SubtractDevicePointers<cuDoubleComplex>(const cuDoubleComplex *d_arr1, const cuDoubleComplex *d_arr2, int N, cuDoubleComplex *out);

template<typename T>
fdsp::GPUArray<T> fdsp::SubtractGPUArrays(const GPUArray<T>& arr1, const GPUArray<T>& arr2)
{
    GPUArray<T> out(arr1.GetDimensionSizes());
    fdsp::SubtractDevicePointers<T>(arr1.GetPointerToArrayConst(), arr2.GetPointerToArrayConst(),
                               arr1.GetSize(), out.GetPointerToArray());
    return out;
}

template fdsp::GPUArray<float> fdsp::SubtractGPUArrays<float>(const GPUArray<float>&, const GPUArray<float>&);
template fdsp::GPUArray<double> fdsp::SubtractGPUArrays<double>(const GPUArray<double>&, const GPUArray<double>&);
template fdsp::GPUArray<cuComplex> fdsp::SubtractGPUArrays<cuComplex>(const GPUArray<cuComplex>&, const GPUArray<cuComplex>&);
template fdsp::GPUArray<cuDoubleComplex> fdsp::SubtractGPUArrays<cuDoubleComplex>(const GPUArray<cuDoubleComplex>&, const GPUArray<cuDoubleComplex>&);

template<typename T>
void fdsp::SubtractGPUArrays(const GPUArray<T>& arr1, const GPUArray<T>& arr2, GPUArray<T>& out)
{
    fdsp::SubtractDevicePointers<T>(arr1.GetPointerToArrayConst(), arr2.GetPointerToArrayConst(),
                               arr1.GetSize(), out.GetPointerToArray());

}

template void fdsp::SubtractGPUArrays<float>(const GPUArray<float>&, const GPUArray<float>&, GPUArray<float>&);
template void fdsp::SubtractGPUArrays<double>(const GPUArray<double>&, const GPUArray<double>&, GPUArray<double>&);
template void fdsp::SubtractGPUArrays<cuComplex>(const GPUArray<cuComplex>&, const GPUArray<cuComplex>&, GPUArray<cuComplex>&);
template void fdsp::SubtractGPUArrays<cuDoubleComplex>(const GPUArray<cuDoubleComplex>&, const GPUArray<cuDoubleComplex>&, GPUArray<cuDoubleComplex>&);

#pragma endregion subtraction

#pragma region multiplication

struct Multiplication
{
    __device__
    float operator()(const float& x, const float& y)
    {
        return x*y;
    }
    __device__
    float operator()(const double& x, const double& y)
    {
        return x*y;
    }
    __device__
    cuComplex operator()(const cuComplex &x, const cuComplex &y)
    {
        cuComplex out;
        out.x = x.x*y.x - x.y*y.y;
        out.y = x.x*y.y + x.y*y.x;
        return out;
    }
    __device__
    cuDoubleComplex operator()(const cuDoubleComplex &x, const cuDoubleComplex &y)
    {
        cuDoubleComplex out;
        out.x = x.x*y.x - x.y*y.y;
        out.y = x.x*y.y + x.y*y.x;
        return out;
    }
};

template<typename T>
void fdsp::MultiplyDevicePointers(const T *d_arr1, const T *d_arr2, int N, T *out)
{
    thrust::transform(thrust::device, d_arr1, d_arr1 + N, d_arr2, out, Multiplication());
}

template void fdsp::MultiplyDevicePointers<float>(const float *d_arr1, const float *d_arr2, int N, float *out);
template void fdsp::MultiplyDevicePointers<double>(const double *d_arr1, const double *d_arr2, int N, double *out);
template void fdsp::MultiplyDevicePointers<cuComplex>(const cuComplex *d_arr1, const cuComplex *d_arr2, int N, cuComplex *out);
template void fdsp::MultiplyDevicePointers<cuDoubleComplex>(const cuDoubleComplex *d_arr1, const cuDoubleComplex *d_arr2, int N, cuDoubleComplex *out);

template<typename T>
fdsp::GPUArray<T> fdsp::MultiplyGPUArrays(const GPUArray<T>& arr1, const GPUArray<T>& arr2)
{
    GPUArray<T> out(arr1.GetDimensionSizes());
    fdsp::MultiplyDevicePointers<T>(arr1.GetPointerToArrayConst(), arr2.GetPointerToArrayConst(),
                                    arr1.GetSize(), out.GetPointerToArray());
    return out;
}

template fdsp::GPUArray<float> fdsp::MultiplyGPUArrays<float>(const GPUArray<float>&, const GPUArray<float>&);
template fdsp::GPUArray<double> fdsp::MultiplyGPUArrays<double>(const GPUArray<double>&, const GPUArray<double>&);
template fdsp::GPUArray<cuComplex> fdsp::MultiplyGPUArrays<cuComplex>(const GPUArray<cuComplex>&, const GPUArray<cuComplex>&);
template fdsp::GPUArray<cuDoubleComplex> fdsp::MultiplyGPUArrays<cuDoubleComplex>(const GPUArray<cuDoubleComplex>&, const GPUArray<cuDoubleComplex>&);

template<typename T>
void fdsp::MultiplyGPUArrays(const GPUArray<T>& arr1, const GPUArray<T>& arr2, GPUArray<T>& out)
{
    fdsp::MultiplyDevicePointers<T>(arr1.GetPointerToArrayConst(), arr2.GetPointerToArrayConst(),
                                    arr1.GetSize(), out.GetPointerToArray());

}

template void fdsp::MultiplyGPUArrays<float>(const GPUArray<float>&, const GPUArray<float>&, GPUArray<float>&);
template void fdsp::MultiplyGPUArrays<double>(const GPUArray<double>&, const GPUArray<double>&, GPUArray<double>&);
template void fdsp::MultiplyGPUArrays<cuComplex>(const GPUArray<cuComplex>&, const GPUArray<cuComplex>&, GPUArray<cuComplex>&);
template void fdsp::MultiplyGPUArrays<cuDoubleComplex>(const GPUArray<cuDoubleComplex>&, const GPUArray<cuDoubleComplex>&, GPUArray<cuDoubleComplex>&);

#pragma endregion

#pragma region division

struct Division
{
    __device__
    float operator()(const float& x, const float& y)
    {
        return x/y;
    }
    __device__
    float operator()(const double& x, const double& y)
    {
        return x/y;
    }
    __device__
    cuComplex operator()(const cuComplex &x, const cuComplex &y)
    {
        cuComplex out;
        float norm = y.x*y.x + y.y*y.y;
        out.x = (x.x*y.x + x.y*y.y)/norm;
        out.y = (x.y*y.x - x.x*y.y)/norm;
        return out;
    }
    __device__
    cuDoubleComplex operator()(const cuDoubleComplex &x, const cuDoubleComplex &y)
    {
        cuDoubleComplex out;
        double norm = y.x*y.x + y.y*y.y;
        out.x = (x.x*y.x + x.y*y.y)/norm;
        out.y = (x.y*y.x - x.x*y.y)/norm;
        return out;
    }
};

template<typename T>
void fdsp::DivideDevicePointers(const T *d_arr1, const T *d_arr2, int N, T *out)
{
    thrust::transform(thrust::device, d_arr1, d_arr1 + N, d_arr2, out, Division());
}

template void fdsp::DivideDevicePointers<float>(const float *d_arr1, const float *d_arr2, int N, float *out);
template void fdsp::DivideDevicePointers<double>(const double *d_arr1, const double *d_arr2, int N, double *out);
template void fdsp::DivideDevicePointers<cuComplex>(const cuComplex *d_arr1, const cuComplex *d_arr2, int N, cuComplex *out);
template void fdsp::DivideDevicePointers<cuDoubleComplex>(const cuDoubleComplex *d_arr1, const cuDoubleComplex *d_arr2, int N, cuDoubleComplex *out);

template<typename T>
fdsp::GPUArray<T> fdsp::DivideGPUArrays(const GPUArray<T>& arr1, const GPUArray<T>& arr2)
{
    GPUArray<T> out(arr1.GetDimensionSizes());
    fdsp::DivideDevicePointers<T>(arr1.GetPointerToArrayConst(), arr2.GetPointerToArrayConst(),
                                    arr1.GetSize(), out.GetPointerToArray());
    return out;
}

template fdsp::GPUArray<float> fdsp::DivideGPUArrays<float>(const GPUArray<float>&, const GPUArray<float>&);
template fdsp::GPUArray<double> fdsp::DivideGPUArrays<double>(const GPUArray<double>&, const GPUArray<double>&);
template fdsp::GPUArray<cuComplex> fdsp::DivideGPUArrays<cuComplex>(const GPUArray<cuComplex>&, const GPUArray<cuComplex>&);
template fdsp::GPUArray<cuDoubleComplex> fdsp::DivideGPUArrays<cuDoubleComplex>(const GPUArray<cuDoubleComplex>&, const GPUArray<cuDoubleComplex>&);

template<typename T>
void fdsp::DivideGPUArrays(const GPUArray<T>& arr1, const GPUArray<T>& arr2, GPUArray<T>& out)
{
    fdsp::DivideDevicePointers<T>(arr1.GetPointerToArrayConst(), arr2.GetPointerToArrayConst(),
                                    arr1.GetSize(), out.GetPointerToArray());

}

template void fdsp::DivideGPUArrays<float>(const GPUArray<float>&, const GPUArray<float>&, GPUArray<float>&);
template void fdsp::DivideGPUArrays<double>(const GPUArray<double>&, const GPUArray<double>&, GPUArray<double>&);
template void fdsp::DivideGPUArrays<cuComplex>(const GPUArray<cuComplex>&, const GPUArray<cuComplex>&, GPUArray<cuComplex>&);
template void fdsp::DivideGPUArrays<cuDoubleComplex>(const GPUArray<cuDoubleComplex>&, const GPUArray<cuDoubleComplex>&, GPUArray<cuDoubleComplex>&);

#pragma endregion
