#pragma once

#include "ir.h"
#include "onnx.pb.h"

namespace onnx { namespace optimization {

void encodeGraph(onnx::ModelProto* p_m, const std::shared_ptr<Graph>& g);

}}
