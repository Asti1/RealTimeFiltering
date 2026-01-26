cmake_minimum_required(VERSION 3.10)
project(Project1)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR /usr/local/opt/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS /opt/homebrew/include /usr/local/include
        PATH_SUFFIXES onnxruntime
)

find_library(ORT_LIB onnxruntime
        PATHS /opt/homebrew/lib /usr/local/lib
)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
    message(FATAL_ERROR "Could not find ONNX Runtime. Check: brew install onnxruntime")
endif()

include_directories(${OpenCV_INCLUDE_DIRS} ${ORT_INCLUDE_DIR})

add_executable(Project1 da2-video.cpp DA2Network.hpp)

target_link_libraries(Project1 ${OpenCV_LIBS} ${ORT_LIB})

