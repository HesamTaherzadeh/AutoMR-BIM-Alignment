cmake_minimum_required(VERSION 3.10)
project(ContextScanManager)

find_package(yaml-cpp REQUIRED)

# PCL setup
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Eigen (header-only)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# OpenCV setup
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/Scancontext
)
# Source files
set(SOURCES
    src/ContextManager/ContextManager.cpp
    src/ScanContext/Scancontext.cpp
)

# Executable
add_executable(main_exe src/main.cpp ${SOURCES})

# Link against PCL, OpenCV
target_link_libraries(main_exe
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    yaml-cpp
)
