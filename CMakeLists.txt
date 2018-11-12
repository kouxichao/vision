cmake_minimum_required(VERSION 3.8)

set (CMAKE_C_COMPILER aarch64-himix100-linux-gcc)#/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-gcc)
set (CMAKE_CXX_COMPILER aarch64-himix100-linux-g++)#/opt/hisi-linux/x86-arm/aarch64-himix100-linux/bin/aarch64-himix100-linux-g++)
#set (CMAKE_FIND_ROOT_PATH /opt/hisi-linux/x86-arm/aarch64-himix100-linux/)

project(demo)
add_subdirectory(src)
