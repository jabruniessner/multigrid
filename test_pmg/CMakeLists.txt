cmake_minimum_required(VERSION 3.16)

project("Testpmgversion")

set(CMAKE_CXX_COMPILER "acpp")



add_library(pmgc SHARED IMPORTED)
set_target_properties(pmgc PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/../pmgc/libpmgc.so"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../pmgc/"
    )

add_library(pmgf SHARED IMPORTED)
set_target_properties(pmgf PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/../pmgf/libmgrid.so"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../pmgf/"
)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS "--acpp-targets=omp.library-only")


add_executable(test_Pbuilder_trilin 
    test_buildP_trilin.cc
)



add_executable(test_Pbuilder_7 
    test_buildP_7.cc
)

add_executable(test_Pbuilder_27
    test_buildP_27.cc
  )

add_executable(test_Gbuilder_1
  test_buildG_1.cc
  )

add_executable(test_Gbuilder_7
  test_buildG_7.cc
  )

add_executable(test_Gbuilder_27
  test_buildG_27.cc
  )

add_executable(test_Bbuilder_7
  test_buildB_7.cc functions.cc
  )


add_executable(test_Bbuilder_27
  test_buildB_27.cc functions.cc 
  )

add_executable(test_gsd7
  test_gsd7.cc functions.cc
  )

add_executable(test_coarsening
  test_coarsening.cc functions.cc
  )

add_executable(test_prolongation
  test_prolongation.cc functions.cc
  )


target_link_libraries(test_prolongation
  PRIVATE
        pmgc
        pmgf
  )


target_link_libraries(test_coarsening
  PRIVATE
        pmgc
        pmgf
  )

target_link_libraries(test_gsd7
  PRIVATE
        pmgc
        pmgf
  )



target_link_libraries(test_Bbuilder_27
  PRIVATE
        pmgc
        pmgf
  )


target_link_libraries(test_Bbuilder_7
  PRIVATE
        pmgc
        pmgf
  )

target_link_libraries(test_Gbuilder_27 
    PRIVATE
        pmgc
        pmgf
)



target_link_libraries(test_Gbuilder_7 
    PRIVATE
        pmgc
        pmgf
)


target_link_libraries(test_Gbuilder_1 
    PRIVATE
        pmgc
        pmgf
)


target_link_libraries(test_Pbuilder_27 
    PRIVATE
        pmgc
        pmgf
)

target_link_libraries(test_Pbuilder_7 
    PRIVATE
        pmgc
        pmgf
)

target_link_libraries(test_Pbuilder_trilin
    PRIVATE
        pmgc
        pmgf
)
