cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mnist_jit)

set(Torch_DIR "/home/john/opt/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

add_executable(mnist_jit mnist_jit.cpp)
target_link_libraries(mnist_jit "${TORCH_LIBRARIES}")
set_property(TARGET mnist_jit PROPERTY CXX_STANDARD 11)
