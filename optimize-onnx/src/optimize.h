#pragma once

#include "ir.h"

namespace onnx { namespace optimization {

std::shared_ptr<Graph> optimize(std::shared_ptr<Graph>, bool init, bool predict);

}}
