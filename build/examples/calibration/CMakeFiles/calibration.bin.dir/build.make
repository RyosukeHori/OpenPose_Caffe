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
include examples/calibration/CMakeFiles/calibration.bin.dir/depend.make

# Include the progress variables for this target.
include examples/calibration/CMakeFiles/calibration.bin.dir/progress.make

# Include the compile flags for this target's objects.
include examples/calibration/CMakeFiles/calibration.bin.dir/flags.make

examples/calibration/CMakeFiles/calibration.bin.dir/calibration.cpp.o: examples/calibration/CMakeFiles/calibration.bin.dir/flags.make
examples/calibration/CMakeFiles/calibration.bin.dir/calibration.cpp.o: ../examples/calibration/calibration.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hori/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/calibration/CMakeFiles/calibration.bin.dir/calibration.cpp.o"
	cd /home/hori/openpose/build/examples/calibration && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/calibration.bin.dir/calibration.cpp.o -c /home/hori/openpose/examples/calibration/calibration.cpp

examples/calibration/CMakeFiles/calibration.bin.dir/calibration.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/calibration.bin.dir/calibration.cpp.i"
	cd /home/hori/openpose/build/examples/calibration && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hori/openpose/examples/calibration/calibration.cpp > CMakeFiles/calibration.bin.dir/calibration.cpp.i

examples/calibration/CMakeFiles/calibration.bin.dir/calibration.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/calibration.bin.dir/calibration.cpp.s"
	cd /home/hori/openpose/build/examples/calibration && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hori/openpose/examples/calibration/calibration.cpp -o CMakeFiles/calibration.bin.dir/calibration.cpp.s

# Object files for target calibration.bin
calibration_bin_OBJECTS = \
"CMakeFiles/calibration.bin.dir/calibration.cpp.o"

# External object files for target calibration.bin
calibration_bin_EXTERNAL_OBJECTS =

examples/calibration/calibration.bin: examples/calibration/CMakeFiles/calibration.bin.dir/calibration.cpp.o
examples/calibration/calibration.bin: examples/calibration/CMakeFiles/calibration.bin.dir/build.make
examples/calibration/calibration.bin: src/openpose/libopenpose.so.1.5.1
examples/calibration/calibration.bin: /usr/local/lib/libopencv_calib3d.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_core.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_dnn.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_features2d.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_flann.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_highgui.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_imgcodecs.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_imgproc.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_ml.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_objdetect.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_photo.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_shape.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_stitching.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_superres.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_video.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_videoio.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_videostab.a
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libglog.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libgflags.so
examples/calibration/calibration.bin: /usr/local/share/OpenCV/3rdparty/lib/liblibprotobuf.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_calib3d.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_features2d.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_flann.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_highgui.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_photo.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_video.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_videoio.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_imgcodecs.a
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libjpeg.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libwebp.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libpng.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libtiff.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libjasper.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libjpeg.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libwebp.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libpng.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libtiff.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libjasper.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libImath.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libIlmImf.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libIex.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libHalf.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libIlmThread.so
examples/calibration/calibration.bin: /usr/local/lib/libopencv_imgproc.a
examples/calibration/calibration.bin: /usr/local/lib/libopencv_core.a
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libz.so
examples/calibration/calibration.bin: /usr/local/share/OpenCV/3rdparty/lib/libittnotify.a
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libcudart_static.a
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/librt.so
examples/calibration/calibration.bin: /usr/lib/x86_64-linux-gnu/libglog.so
examples/calibration/calibration.bin: caffe/lib/libcaffe.so
examples/calibration/calibration.bin: caffe/lib/libcaffe.so
examples/calibration/calibration.bin: examples/calibration/CMakeFiles/calibration.bin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hori/openpose/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable calibration.bin"
	cd /home/hori/openpose/build/examples/calibration && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/calibration.bin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/calibration/CMakeFiles/calibration.bin.dir/build: examples/calibration/calibration.bin

.PHONY : examples/calibration/CMakeFiles/calibration.bin.dir/build

examples/calibration/CMakeFiles/calibration.bin.dir/clean:
	cd /home/hori/openpose/build/examples/calibration && $(CMAKE_COMMAND) -P CMakeFiles/calibration.bin.dir/cmake_clean.cmake
.PHONY : examples/calibration/CMakeFiles/calibration.bin.dir/clean

examples/calibration/CMakeFiles/calibration.bin.dir/depend:
	cd /home/hori/openpose/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hori/openpose /home/hori/openpose/examples/calibration /home/hori/openpose/build /home/hori/openpose/build/examples/calibration /home/hori/openpose/build/examples/calibration/CMakeFiles/calibration.bin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/calibration/CMakeFiles/calibration.bin.dir/depend
