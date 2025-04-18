if(APPLE)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.dylib;.so")
    set(CMAKE_INSTALL_PREFIX "/opt/homebrew")
endif()

cmake_minimum_required(VERSION 3.25)
project(MeshSimplifier)

# 设置C++标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 显式指定Eigen路径
set(EIGEN3_INCLUDE_DIR "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3")

# 查找OpenMesh
find_package(OpenMesh REQUIRED)

find_package(Boost REQUIRED)

# 查找Eigen3（使用我们指定的路径）
find_package(Eigen3 3.4 REQUIRED 
    HINTS ${EIGEN3_INCLUDE_DIR}
    PATHS ${EIGEN3_INCLUDE_DIR}
)

# 打印调试信息
# message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
# message(STATUS "OpenMesh include dir: ${OPENMESH_INCLUDE_DIR}")

set(targetName MeshSimplifier)

add_executable(${targetName} src/main.cpp)

# 添加包含目录
target_include_directories(${targetName}
    PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${OPENMESH_INCLUDE_DIR}
    PUBLIC
    include
)

# 添加链接库
target_link_libraries(${targetName}
    PRIVATE
    OpenMeshCore
    OpenMeshTools
    Eigen3::Eigen
    Boost::boost
)
