# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/server/dk_app/ncnn/examples/face

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/server/dk_app/ncnn/examples/face/build

# Include any dependencies generated for this target.
include src/CMakeFiles/demo_crnn.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/demo_crnn.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/demo_crnn.dir/flags.make

src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o: src/CMakeFiles/demo_crnn.dir/flags.make
src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o: ../src/demo_crnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/server/dk_app/ncnn/examples/face/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o"
	cd /home/server/dk_app/ncnn/examples/face/build/src && /opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o -c /home/server/dk_app/ncnn/examples/face/src/demo_crnn.cpp

src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_crnn.dir/demo_crnn.cpp.i"
	cd /home/server/dk_app/ncnn/examples/face/build/src && /opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/server/dk_app/ncnn/examples/face/src/demo_crnn.cpp > CMakeFiles/demo_crnn.dir/demo_crnn.cpp.i

src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_crnn.dir/demo_crnn.cpp.s"
	cd /home/server/dk_app/ncnn/examples/face/build/src && /opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/server/dk_app/ncnn/examples/face/src/demo_crnn.cpp -o CMakeFiles/demo_crnn.dir/demo_crnn.cpp.s

src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o.requires:

.PHONY : src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o.requires

src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o.provides: src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/demo_crnn.dir/build.make src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o.provides.build
.PHONY : src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o.provides

src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o.provides.build: src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o


# Object files for target demo_crnn
demo_crnn_OBJECTS = \
"CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o"

# External object files for target demo_crnn
demo_crnn_EXTERNAL_OBJECTS =

../bin/demo_crnn: src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o
../bin/demo_crnn: src/CMakeFiles/demo_crnn.dir/build.make
../bin/demo_crnn: ../lib/libncnnface.a
../bin/demo_crnn: /home/server/hisi/hi3559a/P07/Hi3559AV100_SDK_V2.0.0.7/mpp/dk_app/ncnn/build-android-aarch64/src/libncnn.a
../bin/demo_crnn: ../lib/libdlib.a
../bin/demo_crnn: src/CMakeFiles/demo_crnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/server/dk_app/ncnn/examples/face/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/demo_crnn"
	cd /home/server/dk_app/ncnn/examples/face/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo_crnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/demo_crnn.dir/build: ../bin/demo_crnn

.PHONY : src/CMakeFiles/demo_crnn.dir/build

src/CMakeFiles/demo_crnn.dir/requires: src/CMakeFiles/demo_crnn.dir/demo_crnn.cpp.o.requires

.PHONY : src/CMakeFiles/demo_crnn.dir/requires

src/CMakeFiles/demo_crnn.dir/clean:
	cd /home/server/dk_app/ncnn/examples/face/build/src && $(CMAKE_COMMAND) -P CMakeFiles/demo_crnn.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/demo_crnn.dir/clean

src/CMakeFiles/demo_crnn.dir/depend:
	cd /home/server/dk_app/ncnn/examples/face/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/server/dk_app/ncnn/examples/face /home/server/dk_app/ncnn/examples/face/src /home/server/dk_app/ncnn/examples/face/build /home/server/dk_app/ncnn/examples/face/build/src /home/server/dk_app/ncnn/examples/face/build/src/CMakeFiles/demo_crnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/demo_crnn.dir/depend

