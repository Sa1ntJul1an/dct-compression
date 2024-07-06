cmake_minimum_required(VERSION 3.0)
project(DCT-Compression)

# Find OpenCV Package
find_package(OpenCV REQUIRED PATHS C:/opencv/build)

# Include OpenCV Directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Create an Executable
add_executable(DCT-Compression main.cpp)

# Link OpenCV Libraries
target_link_libraries(DCT-Compression ${OpenCV_LIBS})