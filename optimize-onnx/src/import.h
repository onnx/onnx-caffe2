#pragma once

#include "ir.h"
#include "onnx/onnx.pb.h"

namespace onnx { namespace optimization {

std::unique_ptr<Graph> ImportModel(const onnx::ModelProto& mp);

}}
