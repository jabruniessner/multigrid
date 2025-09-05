cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER "acpp")


project(PMGSycl)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SOURCES 
    buildAd.cc
    buildBd.cc
    buildGd.cc
    buildPd.cc
#  cgd.cc
    gsd.cc
  #  matvecd.cc
  #  mgcsd.cc
  #  mgdrvd.cc
  #  mgsubd.cc
  #  mikpckd.cc
    mlinpckd.cc
    #mypdec.cc
    #   newtond.cc
    #   newdrvd.cc
    #   powerd.cc
    #   smoothd.cc
    #   mgfasd.cc
)



set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
add_library(pmgc_double OBJECT ${SOURCES})
target_compile_definitions(pmgc_double PRIVATE USE_DOUBLE)
add_library(pmgc_single OBJECT ${SOURCES})

add_library(pmgc SHARED 
  $<TARGET_OBJECTS:pmgc_single>
  $<TARGET_OBJECTS:pmgc_double>
  )
#target_link_libraries(pmgc PUBLIC pmgc_single pmgc_double)
