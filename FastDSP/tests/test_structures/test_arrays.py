import pytest
import numpy as np

from FastDSP.structures import GPUArray


class TestGPUArray:

    @classmethod
    def setup_class(cls):
        cls.rows = 4
        cls.cols = 6
        cls.array_float = np.ones((cls.rows, cls.cols), dtype=np.float32)
        cls.array_double = np.ones((cls.rows, cls.cols), dtype=np.float64)
        cls.array_float_gpu = GPUArray(cls.array_float)
        cls.array_double_gpu = GPUArray(cls.array_double)

    def test_float_array_transfer_to_device_and_back(self):
        assert(np.all(self.array_float == self.array_float_gpu.get()))

    def test_double_array_transfer_to_device_and_back(self):
        assert(np.all(self.array_double == self.array_double_gpu.get()))

    def test_right_item_is_returned(self):
        for i in range(self.rows):
            for j in range(self.cols):
                assert(self.array_float[i, j] == self.array_float_gpu[i*self.cols + self.rows])
                assert(self.array_double[i, j] == self.array_double_gpu[i*self.cols + self.rows])
