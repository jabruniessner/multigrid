cmake_minimum_required(VERSION 3.10)


project(PMGSycl)

add_library( pmgc
  SHARED
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
