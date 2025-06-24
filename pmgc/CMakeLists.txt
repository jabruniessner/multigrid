cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER "acpp")


project(PMGSycl)

set(SOURCES 
    buildAd.cc
    buildBd.cc
    buildGd.cc
    buildPd.cc
    cgd.cc
    gsd.cc
    matvecd.cc
    mgcsd.cc
    mgdrvd.cc
    #mgsubd.cc
    mikpckd.cc
    mlinpckd.cc
    mypdec.cc
    newtond.cc
    newdrvd.cc
    powerd.cc
    smoothd.cc
    mgfasd.cc
)

set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
add_library(pmgc_double STATIC ${SOURCES})
target_compile_definitions(pmgc_double PRIVATE USE_DOUBLE)
add_library(pmgc_single STATIC ${SOURCES})

add_library(pmgc SHARED combined.cpp)
target_link_libraries(pmgc PRIVATE pmgc_single pmgc_double)
