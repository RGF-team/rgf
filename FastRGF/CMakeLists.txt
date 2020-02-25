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

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.8")
      message(FATAL_ERROR "Insufficient Clang version")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.1.0")
      message(FATAL_ERROR "Insufficient AppleClang version")
    endif()
    cmake_minimum_required(VERSION 3.16)
  endif()
  find_package(OpenMP REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
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
