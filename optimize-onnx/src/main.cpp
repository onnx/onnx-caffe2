#include <iostream>
#include <sstream>

#ifdef _WIN32
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#endif

#include "wrapper.h"

// reads a protobuf file on stdin and writes it back to stdout.
int main(int argc, char** argv) {

  // rudimentary argument parsing
  bool init = (argc == 2 && std::string(argv[1]) == "init");
  bool predict = (argc == 2 && std::string(argv[1]) == "predict");

#ifdef _WIN32
  // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/setmode
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif

  std::stringstream buffer;
  buffer << std::cin.rdbuf();

  std::string content = buffer.str();
  std::string output = onnx::optimization::Optimize(content, init, predict);

  std::cout << output;

  return 0;
}
