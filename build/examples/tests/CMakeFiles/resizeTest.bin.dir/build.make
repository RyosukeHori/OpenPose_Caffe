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
CMAKE_SOURCE_DIR = /home/hori/openpose

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hori/openpose/build

# Include any dependencies generated for this target.
include examples/tests/CMakeFiles/resizeTest.bin.dir/depend.make

# Include the progress variables for this target.
include examples/tests/CMakeFiles/resizeTest.bin.dir/progress.make

# Include the compile flags for this target's objects.
include examples/tests/CMakeFiles/resizeTest.bin.dir/flags.make

examples/tests/CMakeFiles/resizeTest.bin.dir/resizeTest.cpp.o: examples/tests/CMakeFiles/resizeTest.bin.dir/flags.make
examples/tests/CMakeFiles/resizeTest.bin.dir/resizeTest.cpp.o: ../examples/tests/resizeTest.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hori/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/tests/CMakeFiles/resizeTest.bin.dir/resizeTest.cpp.o"
	cd /home/hori/openpose/build/examples/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/resizeTest.bin.dir/resizeTest.cpp.o -c /home/hori/openpose/examples/tests/resizeTest.cpp

examples/tests/CMakeFiles/resizeTest.bin.dir/resizeTest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/resizeTest.bin.dir/resizeTest.cpp.i"
	cd /home/hori/openpose/build/examples/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hori/openpose/examples/tests/resizeTest.cpp > CMakeFiles/resizeTest.bin.dir/resizeTest.cpp.i

examples/tests/CMakeFiles/resizeTest.bin.dir/resizeTest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/resizeTest.bin.dir/resizeTest.cpp.s"
	cd /home/hori/openpose/build/examples/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hori/openpose/examples/tests/resizeTest.cpp -o CMakeFiles/resizeTest.bin.dir/resizeTest.cpp.s

# Object files for target resizeTest.bin
resizeTest_bin_OBJECTS = \
"CMakeFiles/resizeTest.bin.dir/resizeTest.cpp.o"

# External object files for target resizeTest.bin
resizeTest_bin_EXTERNAL_OBJECTS =

examples/tests/resizeTest.bin: examples/tests/CMakeFiles/resizeTest.bin.dir/resizeTest.cpp.o
examples/tests/resizeTest.bin: examples/tests/CMakeFiles/resizeTest.bin.dir/build.make
examples/tests/resizeTest.bin: src/openpose/libopenpose.so.1.5.1
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_calib3d.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_core.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_dnn.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_features2d.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_flann.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_highgui.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_imgcodecs.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_imgproc.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_ml.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_objdetect.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_photo.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_shape.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_stitching.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_superres.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_video.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_videoio.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_videostab.a
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libglog.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libgflags.so
examples/tests/resizeTest.bin: /usr/local/share/OpenCV/3rdparty/lib/liblibprotobuf.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_calib3d.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_features2d.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_flann.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_highgui.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_photo.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_video.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_videoio.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_imgcodecs.a
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libjpeg.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libwebp.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libpng.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libtiff.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libjasper.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libjpeg.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libwebp.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libpng.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libtiff.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libjasper.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libImath.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libIlmImf.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libIex.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libHalf.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libIlmThread.so
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_imgproc.a
examples/tests/resizeTest.bin: /usr/local/lib/libopencv_core.a
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libz.so
examples/tests/resizeTest.bin: /usr/local/share/OpenCV/3rdparty/lib/libittnotify.a
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libcudart_static.a
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/librt.so
examples/tests/resizeTest.bin: /usr/lib/x86_64-linux-gnu/libglog.so
examples/tests/resizeTest.bin: caffe/lib/libcaffe.so
examples/tests/resizeTest.bin: caffe/lib/libcaffe.so
examples/tests/resizeTest.bin: examples/tests/CMakeFiles/resizeTest.bin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hori/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable resizeTest.bin"
	cd /home/hori/openpose/build/examples/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/resizeTest.bin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/tests/CMakeFiles/resizeTest.bin.dir/build: examples/tests/resizeTest.bin

.PHONY : examples/tests/CMakeFiles/resizeTest.bin.dir/build

examples/tests/CMakeFiles/resizeTest.bin.dir/clean:
	cd /home/hori/openpose/build/examples/tests && $(CMAKE_COMMAND) -P CMakeFiles/resizeTest.bin.dir/cmake_clean.cmake
.PHONY : examples/tests/CMakeFiles/resizeTest.bin.dir/clean

examples/tests/CMakeFiles/resizeTest.bin.dir/depend:
	cd /home/hori/openpose/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hori/openpose /home/hori/openpose/examples/tests /home/hori/openpose/build /home/hori/openpose/build/examples/tests /home/hori/openpose/build/examples/tests/CMakeFiles/resizeTest.bin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/tests/CMakeFiles/resizeTest.bin.dir/depend
