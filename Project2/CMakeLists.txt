cmake_minimum_required(VERSION 3.10)
project(Project2)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR /usr/local/opt/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Project2 main.cpp features.cpp features.h csv_util.cpp csv_util.h)

target_link_libraries(Project2 ${OpenCV_LIBS})

