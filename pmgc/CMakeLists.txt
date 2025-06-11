cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_COMPILER "acpp")


project(PMGSycl)

set(SOURCES 
    buildAd.c
    buildBd.c
    buildGd.c
    buildPd.c
    cgd.c
    gsd.c
    matvecd.c
    mgcsd.c
    mgdrvd.c
    mgsubd.c
    mikpckd.c
    mlinpckd.c
    mypdec.c
    newtond.c
    newdrvd.c
    powerd.c
    smoothd.c
    mgfasd.c
)

set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
add_library(pmgc SHARED ${SOURCES})
