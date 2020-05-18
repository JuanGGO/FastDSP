from Cython.Build import cythonize
from distutils.core import Extension, setup
import sys
sys.path.append("../../../")

from FastDSP.core import fdsp_context

if fdsp_context['has_cuda']:

    include_directories = [
        "../../../compiled/cuda/utils/include",
        "../../../compiled/cuda/structures/include",
        "../../../compiled/cuda/core/include",
        fdsp_context['cuda_include_folder']
    ]

    library_directories = [
        "../../../compiled/cuda/cmake-build-release/utils/src",
        "../../../compiled/cuda/cmake-build-release/core/src",
        "../../../compiled/cuda/cmake-build-release/structures/src",
        fdsp_context['cuda_libs_folder']
    ]

    libraries = [
        "cudart",
        "cudadevrt",
        "fdspmath",
        "fdspstructs",
        "fdspinit"
    ]

    sources = [
        "_math_utils.pyx"
    ]

    extension = Extension(
        '_math_utils',
        sources=sources,
        include_dirs=include_directories,
        library_dirs=library_directories,
        libraries=libraries,
        extra_compile_args=['-std=c++14'],
        language='c++'
    )


    setup(
        name='_math_utils',
        version='0.1',
        description='Structures wrapper for cuda and c++ functionality',
        author='Juan Garcia',
        license='MIT',
        ext_modules=cythonize(extension, force=True, include_path=["../../structures/cython"])
    )
