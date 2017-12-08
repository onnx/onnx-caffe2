add_custom_command(
        OUTPUT onnx.pb.h onnx.pb.cc
        COMMAND protoc --proto_path=${CMAKE_SOURCE_DIR}/onnx/onnx --cpp_out=. ${CMAKE_SOURCE_DIR}/onnx/onnx/onnx.proto
        MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/onnx/onnx/onnx.proto
)

add_executable(optimize-onnx
            main.cpp
            assertions.cpp
            export.cpp
            import.cpp
            optimize.cpp
            interned_strings.cpp
            onnx.pb.cc
)

set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")

target_include_directories(optimize-onnx PUBLIC ${CMAKE_SOURCE_DIR}/aten/src)
target_include_directories(optimize-onnx PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_include_directories(optimize-onnx PUBLIC ${CMAKE_BINARY_DIR}/aten/src/ATen)
target_include_directories(optimize-onnx PUBLIC ${CMAKE_BINARY_DIR}/src)
target_link_libraries(     optimize-onnx PUBLIC ATen protobuf)
