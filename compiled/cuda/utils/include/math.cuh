#include "../../structures/include/data_structures.cuh"


namespace fdsp
{

#pragma region arithmetic

#pragma region addition

    template<typename T>
    void AddDevicePointers(const T* d_arr1, const T* d_arr2, int N, T* out);
    template<typename T>
    GPUArray<T> AddGPUArrays(const GPUArray<T>& arr1,const GPUArray<T>& arr2);
    template<typename T>
    void AddGPUArrays(const GPUArray<T>& arr1, const GPUArray<T>& arr2, GPUArray<T>& out);

#pragma endregion

#pragma region subtraction

    template<typename T>
    void SubtractDevicePointers(const T* d_arr1, const T* d_arr2, int N, T* out);
    template<typename T>
    GPUArray<T> SubtractGPUArrays(const GPUArray<T>& arr1,const GPUArray<T>& arr2);
    template<typename T>
    void SubtractGPUArrays(const GPUArray<T>& arr1, const GPUArray<T>& arr2, GPUArray<T>& out);

#pragma endregion

#pragma region multiplication

    template<typename T>
    void MultiplyDevicePointers(const T* d_arr1, const T* d_arr2, int N, T* out);
    template<typename T>
    GPUArray<T> MultiplyGPUArrays(const GPUArray<T>& arr1,const GPUArray<T>& arr2);
    template<typename T>
    void MultiplyGPUArrays(const GPUArray<T>& arr1, const GPUArray<T>& arr2, GPUArray<T>& out);

#pragma endregion

#pragma region division

    template<typename T>
    void DivideDevicePointers(const T* d_arr1, const T* d_arr2, int N, T* out);
    template<typename T>
    GPUArray<T> DivideGPUArrays(const GPUArray<T>& arr1,const GPUArray<T>& arr2);
    template<typename T>
    void DivideGPUArrays(const GPUArray<T>& arr1, const GPUArray<T>& arr2, GPUArray<T>& out);

#pragma endregion

#pragma endregion

}


