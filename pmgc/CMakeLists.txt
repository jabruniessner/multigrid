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
    mgsubd.cc
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
add_library(pmgc SHARED ${SOURCES})
