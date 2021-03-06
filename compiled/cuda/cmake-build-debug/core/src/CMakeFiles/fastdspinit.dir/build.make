# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/juan/Programs/clion-2019.3.3/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/juan/Programs/clion-2019.3.3/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/juan/Documents/Projects/FastDSP/compiled/cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/juan/Documents/Projects/FastDSP/compiled/cuda/cmake-build-debug

# Include any dependencies generated for this target.
include core/src/CMakeFiles/fastdspinit.dir/depend.make

# Include the progress variables for this target.
include core/src/CMakeFiles/fastdspinit.dir/progress.make

# Include the compile flags for this target's objects.
include core/src/CMakeFiles/fastdspinit.dir/flags.make

core/src/CMakeFiles/fastdspinit.dir/initialization.cu.o: core/src/CMakeFiles/fastdspinit.dir/flags.make
core/src/CMakeFiles/fastdspinit.dir/initialization.cu.o: ../core/src/initialization.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/juan/Documents/Projects/FastDSP/compiled/cuda/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object core/src/CMakeFiles/fastdspinit.dir/initialization.cu.o"
	cd /home/juan/Documents/Projects/FastDSP/compiled/cuda/cmake-build-debug/core/src && /usr/local/cuda-10.2/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/juan/Documents/Projects/FastDSP/compiled/cuda/core/src/initialization.cu -o CMakeFiles/fastdspinit.dir/initialization.cu.o

core/src/CMakeFiles/fastdspinit.dir/initialization.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/fastdspinit.dir/initialization.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

core/src/CMakeFiles/fastdspinit.dir/initialization.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/fastdspinit.dir/initialization.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target fastdspinit
fastdspinit_OBJECTS = \
"CMakeFiles/fastdspinit.dir/initialization.cu.o"

# External object files for target fastdspinit
fastdspinit_EXTERNAL_OBJECTS =

core/src/libfastdspinit.a: core/src/CMakeFiles/fastdspinit.dir/initialization.cu.o
core/src/libfastdspinit.a: core/src/CMakeFiles/fastdspinit.dir/build.make
core/src/libfastdspinit.a: core/src/CMakeFiles/fastdspinit.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/juan/Documents/Projects/FastDSP/compiled/cuda/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA static library libfastdspinit.a"
	cd /home/juan/Documents/Projects/FastDSP/compiled/cuda/cmake-build-debug/core/src && $(CMAKE_COMMAND) -P CMakeFiles/fastdspinit.dir/cmake_clean_target.cmake
	cd /home/juan/Documents/Projects/FastDSP/compiled/cuda/cmake-build-debug/core/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fastdspinit.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
core/src/CMakeFiles/fastdspinit.dir/build: core/src/libfastdspinit.a

.PHONY : core/src/CMakeFiles/fastdspinit.dir/build

core/src/CMakeFiles/fastdspinit.dir/clean:
	cd /home/juan/Documents/Projects/FastDSP/compiled/cuda/cmake-build-debug/core/src && $(CMAKE_COMMAND) -P CMakeFiles/fastdspinit.dir/cmake_clean.cmake
.PHONY : core/src/CMakeFiles/fastdspinit.dir/clean

core/src/CMakeFiles/fastdspinit.dir/depend:
	cd /home/juan/Documents/Projects/FastDSP/compiled/cuda/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/juan/Documents/Projects/FastDSP/compiled/cuda /home/juan/Documents/Projects/FastDSP/compiled/cuda/core/src /home/juan/Documents/Projects/FastDSP/compiled/cuda/cmake-build-debug /home/juan/Documents/Projects/FastDSP/compiled/cuda/cmake-build-debug/core/src /home/juan/Documents/Projects/FastDSP/compiled/cuda/cmake-build-debug/core/src/CMakeFiles/fastdspinit.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : core/src/CMakeFiles/fastdspinit.dir/depend

