import numpy as np
from FastDSP.structures import _data_structures


class Complex:

    def __init__(self, real=0, imag=0):
        self.real = real
        self.imag = imag

    def __str__(self):
        return "{} + {}j".format(self.real, self.imag)

