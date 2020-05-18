import pytest
import numpy as np

from FastDSP.structures import GPUArray


@pytest.fixture()
def arrays():
    arrays_gpu = []
    arrays_cpu = []
    arrays_pairs = []
    array = np.random.randint(0, 255, (10, 10, 10), dtype=np.uint8)
    arrays_gpu.append(GPUArray(array))
    arrays_cpu.append(array)
    array = np.random.randint(-1000, 1000, (10, 10, 10), dtype=np.int32)
    arrays_gpu.append(GPUArray(array))
    arrays_cpu.append(array)
    array = np.random.randn(10, 10, 10).astype(np.float32)
    arrays_gpu.append(GPUArray(array))
    arrays_cpu.append(array)
    array = np.random.randn(10, 10, 10)
    arrays_gpu.append(GPUArray(array))
    arrays_cpu.append(array)
    array = np.random.randn(10, 10, 10).astype(np.float32) + 1j*np.random.randn(10, 10, 10).astype(np.float32)
    arrays_gpu.append(GPUArray(array))
    arrays_cpu.append(array)
    array = np.random.randn(10, 10, 10).astype(np.float64) + 1j*np.random.randn(10, 10, 10).astype(np.float64)
    arrays_gpu.append(GPUArray(array))
    arrays_cpu.append(array)

    return zip(arrays_cpu, arrays_gpu)


def test_get_mean_returns_mean_all_types(arrays):
    counter = 0
    for array_cpu, array_gpu in arrays:
        cpu_mean = array_cpu.mean()
        gpu_mean = array_gpu.mean()
        if array_cpu.dtype in [np.complex64, np.complex128]:
            cpu_mean_real = round(cpu_mean.real, 3)
            cpu_mean_imag = round(cpu_mean.imag, 3)
            gpu_mean_real = round(gpu_mean.real, 3)
            gpu_mean_imag = round(gpu_mean.imag, 3)
            assert(pytest.approx(gpu_mean_real) == pytest.approx(cpu_mean_real))
            assert(pytest.approx(cpu_mean_imag) == pytest.approx(gpu_mean_imag))
        else:
            assert(pytest.approx(round(cpu_mean, 3)) == pytest.approx(round(gpu_mean, 3)))
