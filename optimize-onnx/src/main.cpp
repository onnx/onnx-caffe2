#include <fstream>

#ifdef _WIN32
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

#include "export.h"
#include "import.h"

// copied from https://github.com/onnx/onnx/blob/master/onnx/proto_utils.h
template <typename Proto>
bool ParseProtoFromBytes(Proto* proto, const char* buffer, size_t length) {
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  ::google::protobuf::io::CodedInputStream coded_stream(
      new google::protobuf::io::ArrayInputStream(buffer, length));
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

// reads a protobuf file on stdin and writes it back to stdout.
int main(int argc, char** argv) {

#ifdef _WIN32
  // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/setmode
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif

  std::stringstream buffer;
  buffer << std::cin.rdbuf();

  onnx::ModelProto mp_in;
  ParseProtoFromBytes(&mp_in, buffer.str().c_str(), buffer.str().size());
  std::shared_ptr<onnx::optimization::Graph> g = onnx::optimization::ImportModel(mp_in);

  if (g.get() == nullptr) {
    std::cerr << "Warning: optimize-onnx is unable to parse input model" << std::endl;
    std::cout << buffer.str();
  } else {
    onnx::ModelProto mp_out;

    onnx::optimization::encodeGraph(&mp_out, g);

    // output IR version is not dependent on input IR version
    mp_out.set_ir_version(3);
    if (mp_in.has_producer_name()) {
      mp_out.set_producer_name(mp_in.producer_name());
    }
    if (mp_in.has_producer_version()) {
      mp_out.set_producer_version(mp_in.producer_version());
    }
    if (mp_in.has_domain()) {
      mp_out.set_domain(mp_in.domain());
    }
    if (mp_in.has_model_version()) {
      mp_out.set_model_version(mp_in.model_version());
    }
    if (mp_in.has_doc_string()) {
      mp_out.set_doc_string(mp_in.doc_string());
    }
    for (int i = 0; i < mp_in.metadata_props_size(); i++) {
      auto& pp_in = mp_in.metadata_props(i);
      auto* pp_out = mp_out.add_metadata_props();
      if (pp_in.has_key()) {
        pp_out->set_key(pp_in.key());
      }
      if (pp_in.has_value()) {
        pp_out->set_value(pp_in.value());
      }
    }

    std::string out;
    mp_out.SerializeToString(&out);
    std::cout << out;
  }

  return 0;
}
