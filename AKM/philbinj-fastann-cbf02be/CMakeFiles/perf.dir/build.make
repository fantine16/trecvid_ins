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
CMAKE_SOURCE_DIR = /home/liuchang/Desktop/AMK/philbinj-fastann-cbf02be

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liuchang/Desktop/AMK/philbinj-fastann-cbf02be

# Utility rule file for perf.

# Include the progress variables for this target.
include CMakeFiles/perf.dir/progress.make

CMakeFiles/perf: dummy_perf

dummy_perf: perf_dist_l2
dummy_perf: perf_kdtree
	$(CMAKE_COMMAND) -E cmake_progress_report /home/liuchang/Desktop/AMK/philbinj-fastann-cbf02be/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Generating dummy_perf"
	./perf_dist_l2 && ./perf_kdtree

perf: CMakeFiles/perf
perf: dummy_perf
perf: CMakeFiles/perf.dir/build.make
.PHONY : perf

# Rule to build all files generated by this target.
CMakeFiles/perf.dir/build: perf
.PHONY : CMakeFiles/perf.dir/build

CMakeFiles/perf.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/perf.dir/cmake_clean.cmake
.PHONY : CMakeFiles/perf.dir/clean

CMakeFiles/perf.dir/depend:
	cd /home/liuchang/Desktop/AMK/philbinj-fastann-cbf02be && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liuchang/Desktop/AMK/philbinj-fastann-cbf02be /home/liuchang/Desktop/AMK/philbinj-fastann-cbf02be /home/liuchang/Desktop/AMK/philbinj-fastann-cbf02be /home/liuchang/Desktop/AMK/philbinj-fastann-cbf02be /home/liuchang/Desktop/AMK/philbinj-fastann-cbf02be/CMakeFiles/perf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/perf.dir/depend

