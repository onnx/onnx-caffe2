#pragma once

#include "interned_strings.h"
#include "assertions.h"

#include "onnx.pb.h"

#include <memory>
#include <iostream>

namespace onnx { namespace optimization {

struct Dimension {
  Dimension(bool is_int, int64_t dim, std::string param)
    : is_int(is_int), dim(dim), param(std::move(param))
  { }

  bool is_int;
  int64_t dim;
  std::string param;
};

}} // namespace onnx::optimization
