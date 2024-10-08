# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Produce verbose output by default.
VERBOSE = 1

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/nfs/wanghs/hpc/cholesky

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/nfs/wanghs/hpc/cholesky/bulid

# Include any dependencies generated for this target.
include src/CMakeFiles/cholesky.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/cholesky.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/cholesky.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/cholesky.dir/flags.make

src/CMakeFiles/cholesky.dir/cholesky.cu.o: src/CMakeFiles/cholesky.dir/flags.make
src/CMakeFiles/cholesky.dir/cholesky.cu.o: src/CMakeFiles/cholesky.dir/includes_CUDA.rsp
src/CMakeFiles/cholesky.dir/cholesky.cu.o: /mnt/nfs/wanghs/hpc/cholesky/src/cholesky.cu
src/CMakeFiles/cholesky.dir/cholesky.cu.o: src/CMakeFiles/cholesky.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/nfs/wanghs/hpc/cholesky/bulid/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object src/CMakeFiles/cholesky.dir/cholesky.cu.o"
	cd /mnt/nfs/wanghs/hpc/cholesky/bulid/src && /usr/local/cuda-12.5/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT src/CMakeFiles/cholesky.dir/cholesky.cu.o -MF CMakeFiles/cholesky.dir/cholesky.cu.o.d -x cu -c /mnt/nfs/wanghs/hpc/cholesky/src/cholesky.cu -o CMakeFiles/cholesky.dir/cholesky.cu.o

src/CMakeFiles/cholesky.dir/cholesky.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cholesky.dir/cholesky.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

src/CMakeFiles/cholesky.dir/cholesky.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cholesky.dir/cholesky.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cholesky
cholesky_OBJECTS = \
"CMakeFiles/cholesky.dir/cholesky.cu.o"

# External object files for target cholesky
cholesky_EXTERNAL_OBJECTS =

src/cholesky: src/CMakeFiles/cholesky.dir/cholesky.cu.o
src/cholesky: src/CMakeFiles/cholesky.dir/build.make
src/cholesky: src/CMakeFiles/cholesky.dir/linkLibs.rsp
src/cholesky: src/CMakeFiles/cholesky.dir/objects1
src/cholesky: src/CMakeFiles/cholesky.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/nfs/wanghs/hpc/cholesky/bulid/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable cholesky"
	cd /mnt/nfs/wanghs/hpc/cholesky/bulid/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cholesky.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/cholesky.dir/build: src/cholesky
.PHONY : src/CMakeFiles/cholesky.dir/build

src/CMakeFiles/cholesky.dir/clean:
	cd /mnt/nfs/wanghs/hpc/cholesky/bulid/src && $(CMAKE_COMMAND) -P CMakeFiles/cholesky.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/cholesky.dir/clean

src/CMakeFiles/cholesky.dir/depend:
	cd /mnt/nfs/wanghs/hpc/cholesky/bulid && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/nfs/wanghs/hpc/cholesky /mnt/nfs/wanghs/hpc/cholesky/src /mnt/nfs/wanghs/hpc/cholesky/bulid /mnt/nfs/wanghs/hpc/cholesky/bulid/src /mnt/nfs/wanghs/hpc/cholesky/bulid/src/CMakeFiles/cholesky.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/cholesky.dir/depend

