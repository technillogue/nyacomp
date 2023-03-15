#include <pybind11/pybind11.h>
#include "nvcomp/gdeflate.hpp"
#include "nvcomp.hpp" 
#include "nvcomp/nvcompManagerFactory.hpp"

#include <chrono>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <vector>

#include <torch/extension.h>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace nvcomp;
namespace py = pybind11;


#define CUDA_CHECK(cond)                                                       \
  do {                                                                         \
    cudaError_t err = cond;                                                    \
    if (err != cudaSuccess) {                                               \
      std::cerr << "Failure" << std::endl;                                \
      exit(1);                                                              \
    }                                                                         \
  } while (false)

/*
#define TIMER(name, body) \
  do { \
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); \
      body \
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); \
      std::cout << name << " time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl; \
  } while (false)
*/

std::pair<std::vector<uint8_t>, size_t> load_file(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  size_t file_size = static_cast<size_t>(file.tellg());
  std::vector<uint8_t> buffer(file_size);

  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
    throw std::runtime_error("Failed to read file: " + filename);
  }

  return std::make_pair(buffer, file_size);
}


// float compress(const std::string input_filename) {
//   std::vector<uint8_t> uncompressed_data;
//   size_t input_buffer_len;
//   std::tie(uncompressed_data, input_buffer_len) = load_file(filename);
//   std::cout << "read " << input_buffer_len << " bytes from " << filename << std::endl;

int compress(py::bytes pybytes, const std::string filename) {
  std::string bytes_str = pybytes;
  size_t input_buffer_len = bytes_str.size();
  std::vector<uint8_t> uncompressed_data(bytes_str.data(), bytes_str.data() + input_buffer_len );
  std::cout << "working with " << input_buffer_len << " uncompressed bytes" << std::endl;

  uint8_t* device_input_ptrs;
  CUDA_CHECK(cudaMalloc(&device_input_ptrs, input_buffer_len));
  CUDA_CHECK(cudaMemcpy(device_input_ptrs, uncompressed_data.data(), input_buffer_len, cudaMemcpyDefault));

  // start compressing

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  GdeflateManager nvcomp_manager{chunk_size, data_type, stream};
  CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);
  uint8_t* comp_buffer;
  CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));

  // check how long compression takes

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
 
  nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "compression time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  // size of compressed buffer
  size_t comp_size = nvcomp_manager.get_compressed_output_size(comp_buffer);
  // compression ratio
  float comp_ratio = (float)comp_size / (float)input_buffer_len;

  // copy compressed buffer to host memory and then write it to a file
  
  std::ofstream comp_file(filename, std::ios::binary);
  if (!comp_file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::vector<uint8_t> comp_buffer_host(comp_size);
  // std::cout <<"doing cuda memcpy" << std::endl;
  // copy compressed buffer to host memory using stream and syncronize to block for the copy to finish
  CUDA_CHECK(cudaMemcpy(comp_buffer_host.data(), comp_buffer, comp_size, cudaMemcpyDefault));
  std::cout << "writing compressed buffer to file: " << filename << std::endl;
  comp_file.write(reinterpret_cast<const char*>(comp_buffer_host.data()), comp_size);
  comp_file.close();
  

  CUDA_CHECK(cudaFree(comp_buffer));
  CUDA_CHECK(cudaFree(device_input_ptrs));
  CUDA_CHECK(cudaStreamDestroy(stream));

  std::cout << "compressed size: " << comp_size << ", compression ratio: " << comp_ratio << std::endl;

  return comp_size;
}

torch::ScalarType type_for_name(std::string type_name) {
  if (type_name == "uint8") return torch::kUInt8;
   else if (type_name == "int8") return torch::kInt8;
   else if (type_name == "int16") return torch::kInt16;
   else if (type_name == "int32") return torch::kInt32;
   else if (type_name == "int64") return torch::kInt64;
   else if (type_name == "float32") return torch::kFloat32;
   else if (type_name == "float64") return torch::kFloat64;
   else throw std::runtime_error("Unknown type name: " + type_name);
}

// maybe just decompress from an arbitrary stream + length?
// doing an r2 call with a library probably sucks
// can you just get a fd?
// or a python iterator...?
// maybe you can go into python to grab an http library
// 

torch::Tensor decompress(const std::string filename, torch::Tensor tensor) {
  std::vector<uint8_t> compressed_data;
  size_t input_buffer_len;
  std::tie(compressed_data, input_buffer_len) = load_file(filename);
  std::cout << "read " << input_buffer_len << " bytes from " << filename << std::endl;
  std::chrono::steady_clock::time_point copy_begin = std::chrono::steady_clock::now();
  uint8_t* comp_buffer;
  // while file is not eof or length not reached
  // read file into buffer
  // cudaMemcpy 128kb ?
  // i5: L1 90K L2 2MB L3 24MB
  // 3090: L1 128kb L2 6MB
  // check if there's a significant overhead on extra number of copies
  // and either 4kb, 128kb or 2-4MB
  // sd unet is 1.7 GB, vae 580MB, clip 235MB
  CUDA_CHECK(cudaMalloc(&comp_buffer, input_buffer_len));
  CUDA_CHECK(cudaMemcpy(comp_buffer, compressed_data.data(), input_buffer_len, cudaMemcpyDefault));
  std::chrono::steady_clock::time_point copy_end = std::chrono::steady_clock::now();
  std::cout << "copying compressed bytes to gpu took: " << std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_begin).count() << "[µs]" << std::endl;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);

  DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
  std::cout << "decompressing into tensor of size " << decomp_config.decomp_data_size << std::endl;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  // uint8_t* decomp_buffer;
  // CUDA_CHECK(cudaMalloc(&decomp_buffer, decomp_config.decomp_data_size));
  // // auto tensor_options = torch::TensorOptions().dtype(tensor.dtype()).device(torch::kCUDA, 0);
  // size_t element_size = torch::elementSize(torch::typeMetaToScalarType(tensor.dtype()));
  // torch::Tensor decomp_data = torch::empty({decomp_config.decomp_data_size / element_size}, tensor.options());

  // // torch::Tensor decomp_data = torch::empty(tensor.sizes(), torch::TensorOptions().dtype(torch::kUInt8).device(tensor.device()));
  decomp_nvcomp_manager->decompress(static_cast<uint8_t*>(tensor.data_ptr()), comp_buffer, decomp_config);
  // // decomp_nvcomp_manager->decompress(decomp_data.data_ptr<uint8_t>(), comp_buffer, decomp_config);
  // decomp_nvcomp_manager->decompress(decomp_buffer, comp_buffer, decomp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); 
  std::cout << "decompression time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;  // std::cout << "synced" << std::endl;
  // create a new tensor using from_blob by passing the same data_ptr, to reinterpret the same data as the correct dtype
  // torch::Tensor decomp_tensor = torch::from_blob(decomp_buffer, tensor.sizes(), tensor.options());
  // std::cout << "did from_blob" << std::endl;

  // std::cout << "tensor sizes: " << tensor.sizes() << std::endl;
  // std::cout << "decomp_tensor sizes: " << decomp_tensor.sizes() << std::endl;
  // std::cout << "tensor data_ptr: " << tensor.data_ptr() << std::endl;
  // std::cout << "decomp_tensor data_ptr: " << decomp_tensor.data_ptr() << std::endl;
  
  // tensor.copy_(decomp_tensor);
  // tensor.set_(decomp_data);
  // Cast the decompressed data to the correct type before copying it to the output tensor
  // tensor.copy_(decomp_data.to(tensor.dtype()).reshape(tensor.sizes()));
  CUDA_CHECK(cudaFree(comp_buffer));
  // std::cout << "copy made" << std::endl;

  CUDA_CHECK(cudaStreamSynchronize(stream));

  // std::cout << "decompression time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;  // std::cout << "synced" << std::endl;
  CUDA_CHECK(cudaStreamDestroy(stream));
  return tensor;
}

// torch::Tensor& make_tensor() {
//     // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
//     // auto tensor = torch::empty({1024 / sizeof(float)}, options);
//     torch::Tensor tensor = torch::ones(5);
//     // std::cout << "made tensor" << std::endl;
//     return torch::ones(5);;
// }

// std::shared_ptr<torch::Tensor> make_tensor() {
//     auto tensor = std::make_shared<torch::Tensor>(torch::ones(5));
//     std::cout << "made tensor" << std::endl;
//     return tensor;
// }

// torch::Tensor make_tensor() {
//     auto tensor = torch::ones(5);
//     return tensor;
// }

torch::Tensor make_tensor() {
  auto tensor_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0);
  torch::Tensor tensor = torch::empty({100}, tensor_options);
  return tensor;
}

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}


PYBIND11_MODULE(_nyacomp, m) {
    m.doc() = R"pbdoc(python bindings for nvcomp with torch)pbdoc";

    m.def("compress", &compress, R"pbdoc(
        compress
    )pbdoc",  py::arg("data"), py::arg("filename"));

    m.def("decompress", &decompress, R"pbdoc(
        decompress
    )pbdoc",  py::arg("filename"), py::arg("dest_tensor"));

    // m.def("sigmoid", &d_sigmoid, "sigmod fn");

    // m.def("make_tensor", &make_tensor, py::return_value_policy::automatic);

    // m.def("make_tensor", &make_tensor, "make_tensor");

    // m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
    //     Subtract two numbers

    //     Some other explanation about the subtract function.
    // )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
