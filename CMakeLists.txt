# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)



set(CMAKE_CXX_COMPILER "acpp" CACHE STRING "Choose C++ compiler")

# Set the project name
project(MultigridSYCL LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(CMAKE_BUILD_TYPE Release)
#if(NOT DEFINED CMAKE_CXX_COMPILER)
if(CMAKE_CXX_COMPILER MATCHES acpp.*)
set(CMAKE_CXX_FLAGS_RELEASE " --acpp-targets=generic")
set(CMAKE_CXX_FLAGS "--acpp-pcuda")
else()
  set(CMAKE_CUDA_FLAGS "--expt-extended-lambda --expt-relaxed-constexpr")
endif()






include_directories("mdspan")

#find_package(TBB REQUIRED)
#link_libraries(TBB:tbb)

#set(LIB_NAME prof)
#set(LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/profiling_library.cc)
#set(LIB_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lib${LIB_NAME}.so)
#
#
##add_library(${LIB_NAME} SHARED ${LIB_SOURCES})
#
#add_custom_command(
#    OUTPUT ${LIB_OUTPUT}
#    COMMAND nvcc -std=c++20 -Xcompiler -fPIC -shared -o ${LIB_OUTPUT} ${LIB_SOURCES}
#    DEPENDS ${LIB_SOURCES}
#    COMMENT "Building shared library ${LIB_NAME} with custom command"
#)
#
#add_custom_target(${LIB_NAME} ALL DEPENDS ${LIB_OUTPUT})

#endif()

add_library(prof profiling_library.cc)

# Add the executable
add_executable(run_tests tests.cu)
add_executable(test_CG_solver test_CG.cu)

add_executable(test_CG_solver3D test_CG3D.cu)
add_executable(run_test_3D tests3D.cu)
add_executable(test_lvl_trans tests_level_transition.cu)
add_executable(test_lvl_trans3D tests_level_transition3D.cu)
add_executable(MultiGrid_test MultigridDomain_test.cu)
add_executable(Laplacian2D Laplacian_2D.cc)
add_executable(Laplacian3D Laplacian_3D.cc)
add_executable(Laplacian3D_GS Laplacian_3D_GS.cu)



# Link NVTX to your target
add_dependencies(Laplacian3D_GS prof)
target_link_libraries(Laplacian3D_GS PRIVATE prof)
target_compile_definitions(Laplacian3D_GS PRIVATE PROFILING)



add_executable(matrix_creation_test main_test_matrix_creation.cc)
add_executable(test_ILU test_ILU.cc)
add_executable(topological_test dots_in_sphere.cc)
add_executable(dots_ist dots_in_sphere_time_test.cc)
add_executable(pqr_reader_test test_pqr_reader.cc fileio.cc)
add_executable(test_dx test_write_domain_to_dx.cc)
add_executable(create_kappa create_kappa_map.cc fileio.cc)
add_executable(create_charge_test test_charge_add.cc)
add_executable(create_charge_dist create_charge_distribution.cc fileio.cc)
add_executable(test_kappa_functions test_kappa_function.cc)
#add_executable(PBE_example PBE_example.cc fileio.cc)
add_executable(PBE_multigrid_simple PBE_multigrid_simple.cc fileio.cc)
#add_executable(Tetraeda_cut_test cutting_tetrahedra_test.cc)
add_executable(applying_cuts_test applying_cuts_to_domain.cc fileio.cc)
add_executable(applying_cuts_test_debug applying_cuts_to_domain_debug.cc fileio.cc)
add_executable(sort_atoms sort_atoms.cpp fileio.cc)
add_executable(create_dumbell create_dumbell.cc fileio.cc)
add_executable(volume_comp_test test_volume_computation.cc)
add_executable(volume_comp_test_tw test_volume_computation_tw_sphere.cc)
add_executable(compute_vol_2_spheres compute_volume_2_spheres.cc)
add_executable(partial_volume_computes computing_partial_volumes.cc fileio.cc)



message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

# Optional: Link additional libraries if needed
# target_link_libraries(YourExecutableName YourLibraryName)
