cmake_minimum_required(VERSION 3.12...3.18)
project(ecc_python)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_subdirectory(pybind11)
add_subdirectory(EpipolarConsistency)
		
pybind11_add_module(ecc src/main.cpp)

target_include_directories(ecc 
    PRIVATE ${CMAKE_SOURCE_DIR}/EpipolarConsistency/code
    PRIVATE ${CMAKE_SOURCE_DIR}/EpipolarConsistency/code/LibEpipolarConsistency
    PRIVATE ${CMAKE_SOURCE_DIR}/EpipolarConsistency/code/HeaderOnly
    PUBLIC ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(ecc 
    PRIVATE
   LibProjectiveGeometry
   LibUtilsCuda
   LibEpipolarConsistency
   gomp
   Eigen3::Eigen
)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(ecc PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
