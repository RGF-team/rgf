# create a library "base" and test program tester_base

add_library (base utils.cpp data.cpp discretization.cpp classifier.cpp)
#target_include_directories(base PUBLIC ${CMAKE_SOURCE_DIR}/include)

#add_executable(tester_base tester.cpp)
#target_link_libraries(tester_base base)

#install(TARGETS tester_base DESTINATION ${CMAKE_SOURCE_DIR}/src/base)

if(OPENMP)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(base OpenMP::OpenMP_CXX)
  endif()
endif(OPENMP)
