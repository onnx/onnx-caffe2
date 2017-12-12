#pragma once

#include "ir.h"

namespace onnx { namespace optimization {

void optimize(std::shared_ptr<Graph>, bool init, bool predict);

}}
