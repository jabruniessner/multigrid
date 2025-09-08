# Minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(MultigridSYCL)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set(CMAKE_BUILD_TYPE Release)
#if(NOT DEFINED CMAKE_CXX_COMPILER)
#set(CMAKE_CXX_FLAGS_DEBUG "--acpp-targets=omp.library-only -DDEBUGMODE -g -fsanitize=address -fno-omit-frame-pointer")

set(CMAKE_CXX_FLAGS_DEBUG "--acpp-targets=omp.library-only -DDEBUGMODE -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_COMPILER "acpp")

include_directories("mdspan")


include(ExternalProject)

ExternalProject_Add(pmgc
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/pmgc
    BINARY_DIR ${CMAKE_BINARY_DIR}/pmgc
    CMAKE_ARGS 
      -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/pmgc
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
      -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
    INSTALL_COMMAND ""
)


# Paths where pmgc will install
set(PMGC_PREFIX ${CMAKE_BINARY_DIR}/pmgc)
set(PMGC_LIB ${PMGC_PREFIX}/libpmgc.so)   # adjust name for your lib
set(PMGC_INCLUDE ${CMAKE_SOURCE_DIR}/pmgc)

# Create imported target
add_library(pmgc_lib SHARED IMPORTED)
set_target_properties(pmgc_lib PROPERTIES
    IMPORTED_LOCATION ${PMGC_LIB}
    INTERFACE_INCLUDE_DIRECTORIES ${PMGC_INCLUDE}
)


# Make sure it's built before you link against it
add_dependencies(pmgc_lib pmgc)



add_library(prof SHARED profiling_library.cc)

find_package(VTK REQUIRED COMPONENTS
  CommonCore
  CommonDataModel
  IOXML
  ImagingCore
)

link_libraries( 
    VTK::CommonCore
    VTK::CommonDataModel
    VTK::IOXML
    )



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
add_executable(Laplacian3D_GS Laplacian_3D_GS.cc)
# Link NVTX to your target
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
add_executable(PBE_example PBE_example.cc fileio.cc)
add_executable(PBE_multigrid_simple PBE_multigrid_simple.cc fileio.cc)
add_executable(PBE_pmgc_gsd_test PBE_pmgc_gsd_test.cc fileio.cc)
target_link_libraries(PBE_pmgc_gsd_test PRIVATE pmgc_lib)

add_executable(Laplacian_pmgc Laplacian_pmgc.cc fileio.cc)
target_link_libraries(Laplacian_pmgc PRIVATE pmgc_lib)



#add_executable(Tetraeda_cut_test cutting_tetrahedra_test.cc)
add_executable(applying_cuts_test applying_cuts_to_domain.cc fileio.cc)
add_executable(applying_cuts_test_debug applying_cuts_to_domain_debug.cc fileio.cc)
add_executable(sort_atoms sort_atoms.cpp fileio.cc)
add_executable(create_dumbell create_dumbell.cc fileio.cc)
add_executable(volume_comp_test test_volume_computation.cc)
add_executable(volume_comp_test_tw test_volume_computation_tw_sphere.cc)
add_executable(compute_vol_2_spheres compute_volume_2_spheres.cc)
add_executable(partial_volume_computes computing_partial_volumes.cc fileio.cc)


add_executable(Test_epsilon test_epsilon_markes.cc  fileio.cc)
add_executable(get_epsilon Get_epsilons.cc)

add_executable(Laplacian3D_epsilon Laplacian_epsilon_3D.cc  fileio.cc)

add_executable(test_get_epsilon_values Test_get_epsilon_values.cc)


# Link NVTX to your target
target_link_libraries(Laplacian3D_epsilon PRIVATE prof)
#target_compile_definitions(Laplacian3D_epsilon PRIVATE PROFILING)
#target_link_libraries(Laplacian3D_epsilon PRIVATE 
#    VTK::CommonCore
#    VTK::CommonDataModel
#    VTK::IOXML
#    )



message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

# Optional: Link additional libraries if needed
# target_link_libraries(YourExecutableName YourLibraryName)
