cmake_minimum_required(VERSION 3.0)

project (opt-onnx)

SET(NO_CUDA true) # affects aten
add_subdirectory (aten)

subdirs(src)
