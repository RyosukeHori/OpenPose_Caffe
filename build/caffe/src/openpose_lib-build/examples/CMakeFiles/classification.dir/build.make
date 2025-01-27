# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

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


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hori/openpose/3rdparty/caffe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hori/openpose/build/caffe/src/openpose_lib-build

# Include any dependencies generated for this target.
include examples/CMakeFiles/classification.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/classification.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/classification.dir/flags.make

examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o: examples/CMakeFiles/classification.dir/flags.make
examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o: /home/hori/openpose/3rdparty/caffe/examples/cpp_classification/classification.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hori/openpose/build/caffe/src/openpose_lib-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o"
	cd /home/hori/openpose/build/caffe/src/openpose_lib-build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/classification.dir/cpp_classification/classification.cpp.o -c /home/hori/openpose/3rdparty/caffe/examples/cpp_classification/classification.cpp

examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/classification.dir/cpp_classification/classification.cpp.i"
	cd /home/hori/openpose/build/caffe/src/openpose_lib-build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hori/openpose/3rdparty/caffe/examples/cpp_classification/classification.cpp > CMakeFiles/classification.dir/cpp_classification/classification.cpp.i

examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/classification.dir/cpp_classification/classification.cpp.s"
	cd /home/hori/openpose/build/caffe/src/openpose_lib-build/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hori/openpose/3rdparty/caffe/examples/cpp_classification/classification.cpp -o CMakeFiles/classification.dir/cpp_classification/classification.cpp.s

# Object files for target classification
classification_OBJECTS = \
"CMakeFiles/classification.dir/cpp_classification/classification.cpp.o"

# External object files for target classification
classification_EXTERNAL_OBJECTS =

examples/cpp_classification/classification: examples/CMakeFiles/classification.dir/cpp_classification/classification.cpp.o
examples/cpp_classification/classification: examples/CMakeFiles/classification.dir/build.make
examples/cpp_classification/classification: lib/libcaffe.so.1.0.0
examples/cpp_classification/classification: lib/libcaffeproto.a
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libboost_system.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libboost_thread.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libglog.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libgflags.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libprotobuf.so
examples/cpp_classification/classification: /home/hori/anaconda3/lib/libhdf5_cpp.so
examples/cpp_classification/classification: /home/hori/anaconda3/lib/libhdf5.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/librt.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/cpp_classification/classification: /home/hori/anaconda3/lib/libz.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libdl.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libm.so
examples/cpp_classification/classification: /home/hori/anaconda3/lib/libhdf5_hl_cpp.so
examples/cpp_classification/classification: /home/hori/anaconda3/lib/libhdf5_hl.so
examples/cpp_classification/classification: /home/hori/anaconda3/lib/libhdf5_cpp.so
examples/cpp_classification/classification: /home/hori/anaconda3/lib/libhdf5.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/librt.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libpthread.so
examples/cpp_classification/classification: /home/hori/anaconda3/lib/libz.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libdl.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libm.so
examples/cpp_classification/classification: /home/hori/anaconda3/lib/libhdf5_hl_cpp.so
examples/cpp_classification/classification: /home/hori/anaconda3/lib/libhdf5_hl.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libcudart.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libcurand.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libcublas.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libcublas_device.a
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libcudnn.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/liblapack.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libcblas.so
examples/cpp_classification/classification: /usr/lib/x86_64-linux-gnu/libatlas.so
examples/cpp_classification/classification: examples/CMakeFiles/classification.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hori/openpose/build/caffe/src/openpose_lib-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cpp_classification/classification"
	cd /home/hori/openpose/build/caffe/src/openpose_lib-build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/classification.dir/link.txt --verbose=$(VERBOSE)
	cd /home/hori/openpose/build/caffe/src/openpose_lib-build/examples && ln -sf /home/hori/openpose/build/caffe/src/openpose_lib-build/examples/cpp_classification/classification /home/hori/openpose/build/caffe/src/openpose_lib-build/examples/cpp_classification/classification.bin

# Rule to build all files generated by this target.
examples/CMakeFiles/classification.dir/build: examples/cpp_classification/classification

.PHONY : examples/CMakeFiles/classification.dir/build

examples/CMakeFiles/classification.dir/clean:
	cd /home/hori/openpose/build/caffe/src/openpose_lib-build/examples && $(CMAKE_COMMAND) -P CMakeFiles/classification.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/classification.dir/clean

examples/CMakeFiles/classification.dir/depend:
	cd /home/hori/openpose/build/caffe/src/openpose_lib-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hori/openpose/3rdparty/caffe /home/hori/openpose/3rdparty/caffe/examples /home/hori/openpose/build/caffe/src/openpose_lib-build /home/hori/openpose/build/caffe/src/openpose_lib-build/examples /home/hori/openpose/build/caffe/src/openpose_lib-build/examples/CMakeFiles/classification.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/classification.dir/depend

