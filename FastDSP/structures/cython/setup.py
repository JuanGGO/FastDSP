from distutils.core import Extension, setup
from Cython.Build import cythonize
import sys
sys.path.append("../../../")

from FastDSP.core import fdsp_context

if fdsp_context['has_cuda']:
    include_directories = [
        "../../../compiled/cuda/structures/include",
        "../../../compiled/cuda/core/include",
        fdsp_context['cuda_include_folder']
    ]

    library_directories = [
        "../../../compiled/cuda/cmake-build-release/structures/src",
        "../../../compiled/cuda/cmake-build-release/core/src",
        fdsp_context['cuda_libs_folder']
    ]

    libraries = [
        "fdspstructs",
        "cudart",
        "fdspinit"
    ]

    sources = [
        "_data_structures.pyx"
    ]

    extension = Extension(
        '_data_structures',
        sources=sources,
        include_dirs=include_directories,
        library_dirs=library_directories,
        libraries=libraries,
        extra_compile_args=['-std=c++14'],
        language='c++'
    )


    setup(
        name='_data_structures',
        version='0.1',
        description='Structures wrapper for cuda and c++ functionality',
        author='Juan Garcia',
        license='MIT',
        ext_modules=cythonize(extension, force=True)
    )
