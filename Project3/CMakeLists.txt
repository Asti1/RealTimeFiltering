cmake_minimum_required(VERSION 3.10)
project(Project3)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR /usr/local/opt/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Project3 main.cpp)

target_link_libraries(Project3 ${OpenCV_LIBS})

