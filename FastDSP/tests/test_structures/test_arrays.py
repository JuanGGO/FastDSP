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
        cls.array_complex_float = np.ones((cls.rows, cls.cols), dtype=np.complex64)
        cls.array_complex_double = np.ones((cls.rows, cls.cols), dtype=np.complex128)
        cls.array_float_gpu = GPUArray(cls.array_float)
        cls.array_double_gpu = GPUArray(cls.array_double)
        cls.array_float_complex_gpu = GPUArray(cls.array_complex_float)
        cls.array_complex_double_gpu = GPUArray(cls.array_complex_double)

    def test_float_array_transfer_to_device_and_back(self):
        assert(np.all(self.array_float == self.array_float_gpu.get()))

    def test_double_array_transfer_to_device_and_back(self):
        assert(np.all(self.array_double == self.array_double_gpu.get()))

    def test_complex_float_array_transfer_to_device(self):
        assert(np.all(self.array_complex_float == self.array_float_complex_gpu.get()))

    def test_complex_double_array_transfer_to_device(self):
        assert(np.all(self.array_complex_double == self.array_complex_double_gpu.get()))

    def test_right_item_is_returned(self):
        for i in range(self.rows):
            for j in range(self.cols):
                assert(self.array_float[i, j] == self.array_float_gpu[i*self.cols + self.rows])
                assert(self.array_double[i, j] == self.array_double_gpu[i*self.cols + self.rows])
                assert(self.array_complex_float[i, j] == self.array_float_complex_gpu[i*self.cols + self.rows])
                assert(self.array_complex_double[i, j] == self.array_float_complex_gpu[i*self.cols + self.rows])

    def test_float_addition_returns_right_value(self):
        array3_float = self.array_float_gpu + self.array_float_gpu
        array3 = self.array_float + self.array_float
        assert(np.all(array3_float.get() == array3))

    def test_double_addition_returns_right_value(self):
        array3_double = self.array_double_gpu + self.array_double_gpu
        array3 = self.array_double + self.array_double
        assert(np.all(array3_double.get() == array3))

    def test_complex_float_addition_returns_the_right_value(self):
        array3_complex = self.array_float_complex_gpu + self.array_float_complex_gpu
        array3 = self.array_complex_float + self.array_complex_float
        assert(np.all(array3_complex.get() == array3))

    def test_complex_double_addition_return_the_right_value(self):
        array3_complex_double = self.array_complex_double_gpu + self.array_complex_double_gpu
        array3 = self.array_complex_double + self.array_complex_double
        assert(np.all(array3_complex_double.get() == array3))

    def test_float_subtraction_returns_right_value(self):
        array3_float = self.array_float_gpu - self.array_float_gpu
        array3 = self.array_float - self.array_float
        assert(np.all(array3_float.get() == array3))

    def test_double_subtraction_returns_right_value(self):
        array3_double = self.array_double_gpu - self.array_double_gpu
        array3 = self.array_double - self.array_double
        assert(np.all(array3_double.get() == array3))

    def test_complex_float_subtraction_returns_the_right_value(self):
        array3_complex = self.array_float_complex_gpu - self.array_float_complex_gpu
        array3 = self.array_complex_float - self.array_complex_float
        assert(np.all(array3_complex.get() == array3))

    def test_complex_double_subtraction_return_the_right_value(self):
        array3_complex_double = self.array_complex_double_gpu - self.array_complex_double_gpu
        array3 = self.array_complex_double - self.array_complex_double
        assert(np.all(array3_complex_double.get() == array3))

    def test_float_multiplication_returns_right_value(self):
        array3_float = self.array_float_gpu * self.array_float_gpu
        array3 = self.array_float * self.array_float
        assert(np.all(array3_float.get() == array3))

    def test_double_multiplication_returns_right_value(self):
        array3_double = self.array_double_gpu * self.array_double_gpu
        array3 = self.array_double * self.array_double
        assert(np.all(array3_double.get() == array3))

    def test_complex_float_multiplication_returns_the_right_value(self):
        array3_complex = self.array_float_complex_gpu * self.array_float_complex_gpu
        array3 = self.array_complex_float * self.array_complex_float
        assert(np.all(array3_complex.get() == array3))

    def test_complex_double_multiplication_return_the_right_value(self):
        array3_complex_double = self.array_complex_double_gpu * self.array_complex_double_gpu
        array3 = self.array_complex_double * self.array_complex_double
        assert(np.all(array3_complex_double.get() == array3))

    def test_float_division_returns_right_value(self):
        array3_float = self.array_float_gpu / self.array_float_gpu
        array3 = self.array_float / self.array_float
        assert(np.all(array3_float.get() == array3))

    def test_double_division_returns_right_value(self):
        array3_double = self.array_double_gpu / self.array_double_gpu
        array3 = self.array_double / self.array_double
        assert(np.all(array3_double.get() == array3))

    def test_complex_float_division_returns_the_right_value(self):
        array3_complex = self.array_float_complex_gpu / self.array_float_complex_gpu
        array3 = self.array_complex_float / self.array_complex_float
        assert(np.all(array3_complex.get() == array3))

    def test_complex_double_division_return_the_right_value(self):
        array3_complex_double = self.array_complex_double_gpu / self.array_complex_double_gpu
        array3 = self.array_complex_double / self.array_complex_double
        assert(np.all(array3_complex_double.get() == array3))
