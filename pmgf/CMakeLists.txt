cmake_minimum_required(VERSION 3.10)
project(PMG LANGUAGES C Fortran)

# Enable high optimization by default
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 --std=c90")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3")

# Add source directories
include_directories(${CMAKE_SOURCE_DIR})

# Define sources
set(MGRID_SOURCES
    mainsubd.f cgmgdrvd.f newdrvd.f mgdrvd.f powerd.f
    ncgdrvd.f nsordrvd.f ngsdrvd.f nwjdrvd.f nrchdrvd.f
    matvecd.f mikpckd.f smoothd.f nsmoothd.f
    buildAd.f buildBd.f mblasd.f secd.c rand.c
    cgd.f sord.f gsd.f wjd.f rchd.f
    ncgd.f nsord.f ngsd.f nwjd.f nrchd.f
    dlphsubd.f pded.f
    newtond.f cgmgd.f mgcsd.f mgfasd.f mgsubd.f
    mlinpckd.f ninterpd.f buildPd.f buildGd.f
)

file(COPY ${CMAKE_SOURCE_DIR}/in ${CMAKE_SOURCE_DIR}/in_sample
     DESTINATION ${CMAKE_BINARY_DIR})

# Create static library from Fortran sources
add_library(mgrid SHARED ${MGRID_SOURCES})

# Main C and Fortran executables
add_executable(go main.c)
target_link_libraries(go PRIVATE mgrid)

add_executable(gof maind.f)
target_link_libraries(gof PRIVATE mgrid)

