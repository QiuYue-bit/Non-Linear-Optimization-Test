project(NLSSolver)
cmake_minimum_required(VERSION 3.4)

############################################################
# Build Type
############################################################
IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

############################################################
# C++ flags
############################################################

set(CMAKE_CXX_FLAGS "-std=c++11 -g -Wall")
############################################################
# compile output
############################################################
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)  #指明可执行文件的输出保存位置
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)   #指明生成的库文件的输出保存位置

# 输出comiile 命令，方便Understand使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

############### dependencies(找到涉及的各种库) ######################
find_package(Eigen3 REQUIRED)

############### source and test ######################
include_directories(
    include
    utils
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(curveFitting test/curveFitting.cpp src/NLSSolver.cpp)