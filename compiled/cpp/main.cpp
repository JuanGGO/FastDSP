#include <iostream>
#include <data_structures.cuh>

int main() {
    std::vector<float> v = {2, 2, 2, 2};
    std::vector<size_t> size = {2, 2};
    fdsp::GPUArray<float> gpu(v.data(), size);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
