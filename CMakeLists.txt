# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(ConvolutionSYCL)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#if(NOT DEFINED CMAKE_CXX_COMPILER)

set(CMAKE_CXX_FLAGS "-O3")

set(CMAKE_CXX_COMPILER "acpp")

#endif()

# Add the executable
#add_executable(Convolution Convolution2D.cc)
#add_executable(run_tests tests.cc)
#add_executable(test_CG_solver test_CG.cc)

add_executable(test_CG_solver3D test_CG3D.cc)
add_executable(run_test_3D tests3D.cc)

message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

# Optional: Link additional libraries if needed
# target_link_libraries(YourExecutableName YourLibraryName)
