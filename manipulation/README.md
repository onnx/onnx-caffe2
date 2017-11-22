opt-onnx is a library for interacting with ONNX graphs, and a set of optimizations built on top of that library.

It defines an in-memory graph representation with utilities for
convenient manipulation, and functions to convert this representation
to/from ONNX protobuf structs.

It depends on ONNX (for the protobuf definitions) and on ATen (for tensor representation and manipulation).
