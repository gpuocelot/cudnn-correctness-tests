cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(cudnn-correctness-tests CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter)
execute_process(
	COMMAND "${Python3_EXECUTABLE}" -m venv "${CMAKE_CURRENT_BINARY_DIR}/venv"
	COMMAND_ECHO STDOUT)
execute_process(
	COMMAND "${CMAKE_CURRENT_BINARY_DIR}/venv/bin/python" -m pip install torch torchvision torchaudio
	COMMAND_ECHO STDOUT)

file(GLOB_RECURSE CUDNN_INCLUDE "cudnn.h")
get_filename_component(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE} DIRECTORY)
file(GLOB_RECURSE CUDNN_LIBRARIES "libcudnn.so*")

add_executable(test_activation_sample activation_sample/test.cu)
target_include_directories(test_activation_sample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/activation_sample)
target_include_directories(test_activation_sample PRIVATE ${CUDNN_INCLUDE_DIRS})
set_property(TARGET test_activation_sample PROPERTY CXX_STANDARD 14)
target_link_libraries(test_activation_sample ${CUDNN_LIBRARIES})

add_executable(test_conv_sample conv_sample/conv_sample.cu conv_sample/fp16_dev.cu conv_sample/fp16_emu.cu)
target_include_directories(test_conv_sample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/conv_sample)
target_include_directories(test_conv_sample PRIVATE ${CUDNN_INCLUDE_DIRS})
set_property(TARGET test_conv_sample PROPERTY CXX_STANDARD 14)
target_link_libraries(test_conv_sample ${CUDNN_LIBRARIES})

add_executable(test_fcn_sample fcn_sample/test.cu)
target_include_directories(test_fcn_sample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/fcn_sample)
target_include_directories(test_fcn_sample PRIVATE ${CUDNN_INCLUDE_DIRS})
set_property(TARGET test_fcn_sample PROPERTY CXX_STANDARD 14)
target_link_libraries(test_fcn_sample CUDA::cublas ${CUDNN_LIBRARIES})

add_executable(test_pooling_sample pooling_sample/test.cu)
target_include_directories(test_pooling_sample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pooling_sample)
target_include_directories(test_pooling_sample PRIVATE ${CUDNN_INCLUDE_DIRS})
set_property(TARGET test_pooling_sample PROPERTY CXX_STANDARD 14)
target_link_libraries(test_pooling_sample ${CUDNN_LIBRARIES})

add_executable(test_softmax_sample softmax_sample/test.cu)
target_include_directories(test_softmax_sample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/softmax_sample)
target_include_directories(test_softmax_sample PRIVATE ${CUDNN_INCLUDE_DIRS})
set_property(TARGET test_softmax_sample PROPERTY CXX_STANDARD 14)
target_link_libraries(test_softmax_sample ${CUDNN_LIBRARIES})

include(CTest)

add_test(NAME test_activation_sample COMMAND $<TARGET_FILE:test_activation_sample>)
add_test(NAME test_conv_sample COMMAND $<TARGET_FILE:test_conv_sample>)
add_test(NAME test_fcn_sample COMMAND $<TARGET_FILE:test_fcn_sample>)
add_test(NAME test_pooling_sample COMMAND $<TARGET_FILE:test_pooling_sample>)
add_test(NAME test_softmax_sample COMMAND $<TARGET_FILE:test_softmax_sample>)

