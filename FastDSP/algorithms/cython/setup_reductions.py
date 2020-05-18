from Cython.Build import cythonize
from distutils.core import Extension, setup
import sys
sys.path.append("../../../")

from FastDSP.core import fdsp_context

if fdsp_context['has_cuda']:

    include_directories = [
        "../../../compiled/cuda/algorithms/include",
        "../../../compiled/cuda/structures/include",
        "../../../compiled/cuda/core/include",
        fdsp_context['cuda_include_folder']
    ]

    library_directories = [
        "../../../compiled/cuda/cmake-build-release/algorithms/src",
        "../../../compiled/cuda/cmake-build-release/core/src",
        "../../../compiled/cuda/cmake-build-release/structures/src",
        fdsp_context['cuda_libs_folder']
    ]

    libraries = [
        "cudart",
        "cudadevrt",
        "fdspalgorithms",
        "fdspstructs",
        "fdspinit"
    ]

    sources = [
        "_reductions.pyx"
    ]

    extension = Extension(
        '_reductions',
        sources=sources,
        include_dirs=include_directories,
        library_dirs=library_directories,
        libraries=libraries,
        extra_compile_args=['-std=c++14'],
        language='c++'
    )


    setup(
        name='_reductions',
        version='0.1',
        description='Reductions operation wrapper for cuda and c++ functionality',
        author='Juan Garcia',
        license='MIT',
        ext_modules=cythonize(extension, force=True, include_path=["../../structures/cython"])
    )
