import os

from tools.python.file_manipulation import find_in_path


def locate_cuda():
    nvcc = find_in_path('nvcc', os.environ['PATH'])  # looking for nvidia compiler
    if nvcc is not None:
        return nvcc[:-8]
    return None


def get_cuda_include_and_lib_folders():
    """
    Locates the cuda include folders and lib
    """
    cuda_location = locate_cuda()
    if cuda_location is None:
        print("cuda installation cannot be found. Please make sure cuda is install or set the"
              "environment variable CUDAHOME with the path to cuda installation. FastDSP cuda-enabled"
              "capabilities won't be available")
        return None

    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
    elif 'CUDAPATH' in os.environ:
        home = os.environ['CUDAPATH']
    else:
        home = cuda_location

    cudaconfig = {'home': home, 'include': os.path.join(home, 'include'), 'lib64': os.path.join(home, 'lib64')}

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig
