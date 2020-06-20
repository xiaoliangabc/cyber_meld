cmake_minimum_required(VERSION 2.8.3)
project(ego_lane_detection)

## Compile as C++11
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(PCL REQUIRED )
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

## Specify additional locations of header files
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

## Specify build type
set(CMAKE_BUILD_TYPE "Release")

## Declare a C++ library
add_executable(ego_lane_detection
  src/main.cc
  src/common_utils.cc
  src/roi_selection.cc
  src/feature_extraction.cc
  src/parameter_estimation.cc
  src/lane_segmentation.cc
)

## Specify libraries to link a library or executable target against
target_link_libraries(ego_lane_detection ${PCL_LIBRARIES})
target_link_libraries(ego_lane_detection ${OpenCV_LIBRARIES})
target_link_libraries(ego_lane_detection ${PROJECT_SOURCE_DIR}/lib/libfade2d.so)
