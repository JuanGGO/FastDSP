#!/bin/bash

: '
This script is to compile all cython modules in
the right order to respect binaries dependencies
'

work_dir=`pwd`
# core
cd ../../FastDSP/core/cython/
python setup.py build_ext -i
cd $work_dir

# structures
cd ../../FastDSP/structures/cython/
python setup.py build_ext -i
cd $work_dir

# algorithms
cd ../../FastDSP/algorithms/cython/
python setup_reductions.py build_ext -i
cd $work_dir

# utils
cd ../../FastDSP/utils/cython/
python setup.py build_ext -i
cd $work_dir


