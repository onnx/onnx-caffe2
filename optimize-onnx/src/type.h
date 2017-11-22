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

inline std::vector<Dimension> sizeToDimensions(at::ArrayRef<int64_t> size) {
  std::vector<Dimension> dims;
  dims.reserve(size.size());
  for (auto s : size) {
    dims.push_back(Dimension(true, s, ""));
  }
  return dims;
}

}} // namespace onnx::optimization
