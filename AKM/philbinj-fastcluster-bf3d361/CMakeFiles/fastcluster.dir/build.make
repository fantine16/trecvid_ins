# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361

# Include any dependencies generated for this target.
include CMakeFiles/fastcluster.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fastcluster.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fastcluster.dir/flags.make

CMakeFiles/fastcluster.dir/mpi_queue.cpp.o: CMakeFiles/fastcluster.dir/flags.make
CMakeFiles/fastcluster.dir/mpi_queue.cpp.o: mpi_queue.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/fastcluster.dir/mpi_queue.cpp.o"
	mpic++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/fastcluster.dir/mpi_queue.cpp.o -c /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/mpi_queue.cpp

CMakeFiles/fastcluster.dir/mpi_queue.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fastcluster.dir/mpi_queue.cpp.i"
	mpic++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/mpi_queue.cpp > CMakeFiles/fastcluster.dir/mpi_queue.cpp.i

CMakeFiles/fastcluster.dir/mpi_queue.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fastcluster.dir/mpi_queue.cpp.s"
	mpic++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/mpi_queue.cpp -o CMakeFiles/fastcluster.dir/mpi_queue.cpp.s

CMakeFiles/fastcluster.dir/mpi_queue.cpp.o.requires:
.PHONY : CMakeFiles/fastcluster.dir/mpi_queue.cpp.o.requires

CMakeFiles/fastcluster.dir/mpi_queue.cpp.o.provides: CMakeFiles/fastcluster.dir/mpi_queue.cpp.o.requires
	$(MAKE) -f CMakeFiles/fastcluster.dir/build.make CMakeFiles/fastcluster.dir/mpi_queue.cpp.o.provides.build
.PHONY : CMakeFiles/fastcluster.dir/mpi_queue.cpp.o.provides

CMakeFiles/fastcluster.dir/mpi_queue.cpp.o.provides.build: CMakeFiles/fastcluster.dir/mpi_queue.cpp.o

CMakeFiles/fastcluster.dir/kmeans.cpp.o: CMakeFiles/fastcluster.dir/flags.make
CMakeFiles/fastcluster.dir/kmeans.cpp.o: kmeans.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/fastcluster.dir/kmeans.cpp.o"
	mpic++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/fastcluster.dir/kmeans.cpp.o -c /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/kmeans.cpp

CMakeFiles/fastcluster.dir/kmeans.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fastcluster.dir/kmeans.cpp.i"
	mpic++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/kmeans.cpp > CMakeFiles/fastcluster.dir/kmeans.cpp.i

CMakeFiles/fastcluster.dir/kmeans.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fastcluster.dir/kmeans.cpp.s"
	mpic++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/kmeans.cpp -o CMakeFiles/fastcluster.dir/kmeans.cpp.s

CMakeFiles/fastcluster.dir/kmeans.cpp.o.requires:
.PHONY : CMakeFiles/fastcluster.dir/kmeans.cpp.o.requires

CMakeFiles/fastcluster.dir/kmeans.cpp.o.provides: CMakeFiles/fastcluster.dir/kmeans.cpp.o.requires
	$(MAKE) -f CMakeFiles/fastcluster.dir/build.make CMakeFiles/fastcluster.dir/kmeans.cpp.o.provides.build
.PHONY : CMakeFiles/fastcluster.dir/kmeans.cpp.o.provides

CMakeFiles/fastcluster.dir/kmeans.cpp.o.provides.build: CMakeFiles/fastcluster.dir/kmeans.cpp.o

CMakeFiles/fastcluster.dir/randomkit.c.o: CMakeFiles/fastcluster.dir/flags.make
CMakeFiles/fastcluster.dir/randomkit.c.o: randomkit.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/fastcluster.dir/randomkit.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/fastcluster.dir/randomkit.c.o   -c /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/randomkit.c

CMakeFiles/fastcluster.dir/randomkit.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/fastcluster.dir/randomkit.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/randomkit.c > CMakeFiles/fastcluster.dir/randomkit.c.i

CMakeFiles/fastcluster.dir/randomkit.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/fastcluster.dir/randomkit.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/randomkit.c -o CMakeFiles/fastcluster.dir/randomkit.c.s

CMakeFiles/fastcluster.dir/randomkit.c.o.requires:
.PHONY : CMakeFiles/fastcluster.dir/randomkit.c.o.requires

CMakeFiles/fastcluster.dir/randomkit.c.o.provides: CMakeFiles/fastcluster.dir/randomkit.c.o.requires
	$(MAKE) -f CMakeFiles/fastcluster.dir/build.make CMakeFiles/fastcluster.dir/randomkit.c.o.provides.build
.PHONY : CMakeFiles/fastcluster.dir/randomkit.c.o.provides

CMakeFiles/fastcluster.dir/randomkit.c.o.provides.build: CMakeFiles/fastcluster.dir/randomkit.c.o

CMakeFiles/fastcluster.dir/whetstone.cpp.o: CMakeFiles/fastcluster.dir/flags.make
CMakeFiles/fastcluster.dir/whetstone.cpp.o: whetstone.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/fastcluster.dir/whetstone.cpp.o"
	mpic++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/fastcluster.dir/whetstone.cpp.o -c /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/whetstone.cpp

CMakeFiles/fastcluster.dir/whetstone.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fastcluster.dir/whetstone.cpp.i"
	mpic++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/whetstone.cpp > CMakeFiles/fastcluster.dir/whetstone.cpp.i

CMakeFiles/fastcluster.dir/whetstone.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fastcluster.dir/whetstone.cpp.s"
	mpic++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/whetstone.cpp -o CMakeFiles/fastcluster.dir/whetstone.cpp.s

CMakeFiles/fastcluster.dir/whetstone.cpp.o.requires:
.PHONY : CMakeFiles/fastcluster.dir/whetstone.cpp.o.requires

CMakeFiles/fastcluster.dir/whetstone.cpp.o.provides: CMakeFiles/fastcluster.dir/whetstone.cpp.o.requires
	$(MAKE) -f CMakeFiles/fastcluster.dir/build.make CMakeFiles/fastcluster.dir/whetstone.cpp.o.provides.build
.PHONY : CMakeFiles/fastcluster.dir/whetstone.cpp.o.provides

CMakeFiles/fastcluster.dir/whetstone.cpp.o.provides.build: CMakeFiles/fastcluster.dir/whetstone.cpp.o

CMakeFiles/fastcluster.dir/mpi_utils.cpp.o: CMakeFiles/fastcluster.dir/flags.make
CMakeFiles/fastcluster.dir/mpi_utils.cpp.o: mpi_utils.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/fastcluster.dir/mpi_utils.cpp.o"
	mpic++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/fastcluster.dir/mpi_utils.cpp.o -c /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/mpi_utils.cpp

CMakeFiles/fastcluster.dir/mpi_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fastcluster.dir/mpi_utils.cpp.i"
	mpic++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/mpi_utils.cpp > CMakeFiles/fastcluster.dir/mpi_utils.cpp.i

CMakeFiles/fastcluster.dir/mpi_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fastcluster.dir/mpi_utils.cpp.s"
	mpic++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/mpi_utils.cpp -o CMakeFiles/fastcluster.dir/mpi_utils.cpp.s

CMakeFiles/fastcluster.dir/mpi_utils.cpp.o.requires:
.PHONY : CMakeFiles/fastcluster.dir/mpi_utils.cpp.o.requires

CMakeFiles/fastcluster.dir/mpi_utils.cpp.o.provides: CMakeFiles/fastcluster.dir/mpi_utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/fastcluster.dir/build.make CMakeFiles/fastcluster.dir/mpi_utils.cpp.o.provides.build
.PHONY : CMakeFiles/fastcluster.dir/mpi_utils.cpp.o.provides

CMakeFiles/fastcluster.dir/mpi_utils.cpp.o.provides.build: CMakeFiles/fastcluster.dir/mpi_utils.cpp.o

# Object files for target fastcluster
fastcluster_OBJECTS = \
"CMakeFiles/fastcluster.dir/mpi_queue.cpp.o" \
"CMakeFiles/fastcluster.dir/kmeans.cpp.o" \
"CMakeFiles/fastcluster.dir/randomkit.c.o" \
"CMakeFiles/fastcluster.dir/whetstone.cpp.o" \
"CMakeFiles/fastcluster.dir/mpi_utils.cpp.o"

# External object files for target fastcluster
fastcluster_EXTERNAL_OBJECTS =

libfastcluster.so: CMakeFiles/fastcluster.dir/mpi_queue.cpp.o
libfastcluster.so: CMakeFiles/fastcluster.dir/kmeans.cpp.o
libfastcluster.so: CMakeFiles/fastcluster.dir/randomkit.c.o
libfastcluster.so: CMakeFiles/fastcluster.dir/whetstone.cpp.o
libfastcluster.so: CMakeFiles/fastcluster.dir/mpi_utils.cpp.o
libfastcluster.so: CMakeFiles/fastcluster.dir/build.make
libfastcluster.so: CMakeFiles/fastcluster.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library libfastcluster.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fastcluster.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fastcluster.dir/build: libfastcluster.so
.PHONY : CMakeFiles/fastcluster.dir/build

CMakeFiles/fastcluster.dir/requires: CMakeFiles/fastcluster.dir/mpi_queue.cpp.o.requires
CMakeFiles/fastcluster.dir/requires: CMakeFiles/fastcluster.dir/kmeans.cpp.o.requires
CMakeFiles/fastcluster.dir/requires: CMakeFiles/fastcluster.dir/randomkit.c.o.requires
CMakeFiles/fastcluster.dir/requires: CMakeFiles/fastcluster.dir/whetstone.cpp.o.requires
CMakeFiles/fastcluster.dir/requires: CMakeFiles/fastcluster.dir/mpi_utils.cpp.o.requires
.PHONY : CMakeFiles/fastcluster.dir/requires

CMakeFiles/fastcluster.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fastcluster.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fastcluster.dir/clean

CMakeFiles/fastcluster.dir/depend:
	cd /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361 /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361 /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361 /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361 /home/liuchang/Desktop/AMK/philbinj-fastcluster-bf3d361/CMakeFiles/fastcluster.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fastcluster.dir/depend
