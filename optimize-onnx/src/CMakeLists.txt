add_custom_target(create-onnx-dir ALL COMMAND ${CMAKE_COMMAND} -E make_directory onnx)

add_custom_command(
        OUTPUT onnx/onnx.pb.h onnx/onnx.pb.cc
        COMMAND protoc --proto_path=${CMAKE_SOURCE_DIR}/onnx/onnx --cpp_out=onnx ${CMAKE_SOURCE_DIR}/onnx/onnx/onnx.proto
        DEPENDS create-onnx-dir
        MAIN_DEPENDENCY ${CMAKE_SOURCE_DIR}/onnx/onnx/onnx.proto
)

add_executable(optimize-onnx
            main.cpp
            assertions.cpp
            export.cpp
            import.cpp
            optimize.cpp
            interned_strings.cpp
            onnx/onnx.pb.cc
            wrapper.cpp
)

set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")

target_include_directories(optimize-onnx PUBLIC ${CMAKE_SOURCE_DIR}/aten/src)
target_include_directories(optimize-onnx PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_include_directories(optimize-onnx PUBLIC ${CMAKE_BINARY_DIR}/aten/src/ATen)
target_include_directories(optimize-onnx PUBLIC ${CMAKE_BINARY_DIR}/src)
target_link_libraries(     optimize-onnx PUBLIC ATen protobuf)
