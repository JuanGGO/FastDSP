from distutils.core import Extension, setup
from Cython.Build import cythonize


include_directories = [
    "../../../compiled/cuda/core/include"
]

#TODO: Automatize the search of cuda libraries
library_directories = [
    "../../../compiled/cuda/cmake-build-release/core/src",
    "/usr/local/cuda-10.2/lib64"
]

libraries = [
    "fdspinit",
    "cudart"
]

sources = [
    "_core.pyx"
]

extension = Extension(
    '_core',
    sources=sources,
    include_dirs=include_directories,
    library_dirs=library_directories,
    libraries=libraries,
    extra_compile_args=['-std=c++14'],
    language='c++'
)


setup(
    name='_core',
    version='0.1',
    description='Core wrapper for cuda and c++ functionality',
    author='Juan Garcia',
    license='MIT',
    ext_modules=cythonize(extension, force=True)
)
