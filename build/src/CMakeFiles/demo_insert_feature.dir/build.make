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
include src/CMakeFiles/demo_insert_feature.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/demo_insert_feature.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/demo_insert_feature.dir/flags.make

src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o: src/CMakeFiles/demo_insert_feature.dir/flags.make
src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o: ../src/demo_insert_feature.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/server/dk_app/ncnn/examples/face/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o"
	cd /home/server/dk_app/ncnn/examples/face/build/src && /opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o -c /home/server/dk_app/ncnn/examples/face/src/demo_insert_feature.cpp

src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.i"
	cd /home/server/dk_app/ncnn/examples/face/build/src && /opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/server/dk_app/ncnn/examples/face/src/demo_insert_feature.cpp > CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.i

src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.s"
	cd /home/server/dk_app/ncnn/examples/face/build/src && /opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/server/dk_app/ncnn/examples/face/src/demo_insert_feature.cpp -o CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.s

src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o.requires:

.PHONY : src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o.requires

src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o.provides: src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/demo_insert_feature.dir/build.make src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o.provides.build
.PHONY : src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o.provides

src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o.provides.build: src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o


# Object files for target demo_insert_feature
demo_insert_feature_OBJECTS = \
"CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o"

# External object files for target demo_insert_feature
demo_insert_feature_EXTERNAL_OBJECTS =

../bin/demo_insert_feature: src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o
../bin/demo_insert_feature: src/CMakeFiles/demo_insert_feature.dir/build.make
../bin/demo_insert_feature: ../lib/libncnnface.a
../bin/demo_insert_feature: ../lib/libncnn.a
../bin/demo_insert_feature: ../lib/libsqlite3.a
../bin/demo_insert_feature: ../lib/libdlib.a
../bin/demo_insert_feature: src/CMakeFiles/demo_insert_feature.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/server/dk_app/ncnn/examples/face/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/demo_insert_feature"
	cd /home/server/dk_app/ncnn/examples/face/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo_insert_feature.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/demo_insert_feature.dir/build: ../bin/demo_insert_feature

.PHONY : src/CMakeFiles/demo_insert_feature.dir/build

src/CMakeFiles/demo_insert_feature.dir/requires: src/CMakeFiles/demo_insert_feature.dir/demo_insert_feature.cpp.o.requires

.PHONY : src/CMakeFiles/demo_insert_feature.dir/requires

src/CMakeFiles/demo_insert_feature.dir/clean:
	cd /home/server/dk_app/ncnn/examples/face/build/src && $(CMAKE_COMMAND) -P CMakeFiles/demo_insert_feature.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/demo_insert_feature.dir/clean

src/CMakeFiles/demo_insert_feature.dir/depend:
	cd /home/server/dk_app/ncnn/examples/face/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/server/dk_app/ncnn/examples/face /home/server/dk_app/ncnn/examples/face/src /home/server/dk_app/ncnn/examples/face/build /home/server/dk_app/ncnn/examples/face/build/src /home/server/dk_app/ncnn/examples/face/build/src/CMakeFiles/demo_insert_feature.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/demo_insert_feature.dir/depend

