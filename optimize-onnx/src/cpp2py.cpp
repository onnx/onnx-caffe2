#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "wrapper.h"

namespace onnx { namespace optimization {

namespace py = pybind11;

PYBIND11_MODULE(cpp2py, m) {
  m.doc() = "Python interface to onnx";
  m.def(
      "optimize", [](std::string& content, bool init, bool predict) {
          return py::bytes(Optimize(content, init, predict));
      });
}

}} // namespace onnx::optimization
