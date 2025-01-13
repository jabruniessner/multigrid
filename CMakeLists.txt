# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MultigridSYCL)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)
#if(NOT DEFINED CMAKE_CXX_COMPILER)
set(CMAKE_CXX_FLAGS_DEBUG "--acpp-targets=omp.library-only -DDEBUGMODE -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "-L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include")

set(CMAKE_CXX_COMPILER "acpp")

#endif()

# Add the executable
add_executable(run_tests tests.cc)
add_executable(test_CG_solver test_CG.cc)

add_executable(test_CG_solver3D test_CG3D.cc)
add_executable(run_test_3D tests3D.cc)
add_executable(test_lvl_trans tests_level_transition.cc)
add_executable(test_lvl_trans3D tests_level_transition3D.cc)
add_executable(MultiGrid_test MultigridDomain_test.cc)
add_executable(Laplacian2D Laplacian_2D.cc)
add_executable(Laplacian3D Laplacian_3D.cc)
add_executable(cuda_amg_test amg_solver.cc)

message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

# Optional: Link additional libraries if needed
# target_link_libraries(YourExecutableName YourLibraryName)
