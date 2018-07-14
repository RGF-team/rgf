cmake_minimum_required (VERSION 2.8.0)

project (FastRGF) 

# whether to use OpenMP (default is ON)
option(OPENMP "Use OpenMP for multithreading" ON)

set(CMAKE_CXX_FLAGS "-O3 -std=c++11")
#set(CMAKE_CXX_FLAGS "-g -std=c++11 -Wall")


if(OPENMP)
 message("Use OpenMP for multithreading")
 # use OpenMP
 add_definitions("-DUSE_OMP")		

  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    #  macOS: Clang does not support OpenMP, need to install a g++ with OpenMP support
    #  if the executables are gcc-7 and g++-7, then use the following in build/
    #  cmake .. -DCMAKE_C_COMPILER=gcc-7 -DCMAKE_CXX_COMPILER=g++-7
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "[gG][nN][uU]")
     message(FATAL_ERROR "Compilation only with g++ is supported.\nInstall a gcc-x and g++-x and use cmake command option -DCMAKE_C_COMPILER=gcc-x -DCMAKE_CXX_COMPILER=g++-x, where x is a version number")
    endif()
    set(CMAKE_CXX_COMPILER_ID "GNU")
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
else()
 message("Use standard C++11 thread library")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -ftree-vectorize -ffast-math")
endif()

if(WIN32 AND MINGW)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif()

message("C++ compiler: " ${CMAKE_CXX_COMPILER})
message("C++ options: " ${CMAKE_CXX_FLAGS})
get_directory_property(cDirDefs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
message("C++ definitions: " ${cDirDefs})

include_directories(include)

add_subdirectory(src/base)
add_subdirectory(src/forest)
add_subdirectory(src/exe)
