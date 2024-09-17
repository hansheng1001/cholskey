cmake_minimum_required(VERSION 3.12)


set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -DDEBUG")
set(CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O0 -Wall -g -ggdb -DDEBUG")



# 使用下面这种方法才能够在编译中加入-G选项
# add_compile_options("-G")


message("${CMAKE_CXX_FLAGS_DEBUG}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
# set(CUDA_ARCHITECTURES 80)
# set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_ARCHITECTURES 86)

#H100 90
#2880Ti 75
#4090 86
# A800 80

# set(CMAKE_CUDA_ARCHITECTURES 86) # for 4090

# 在2.ndsl和1.ndsl中找不到编译器,需要进行设置.在87上要注释掉
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.5/bin/nvcc)
project(myqrcode LANGUAGES CXX CUDA)

# 设置可以打印出核函数使用寄存器数和大小数的函数
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -Xcompiler -rdynamic -lineinfo")

add_subdirectory(src)