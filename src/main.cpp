#include <pybind11/pybind11.h>
#include "nvcomp/gdeflate.hpp"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

#include <thread>
#include <future>

#include <chrono>
#include <assert.h>
#include <mutex>
#include <iostream>
#include <fstream>
#include <vector>

#include <torch/extension.h>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace nvcomp;
namespace py = pybind11;

#define CUDA_CHECK(cond)                                                   \
  do                                                                       \
  {                                                                        \
    cudaError_t err = cond;                                                \
    if (err != cudaSuccess)                                                \
    {                                                                      \
      std::cerr << "Cuda error: " << cudaGetErrorString(err) << std::endl; \
      exit(1);                                                             \
    }                                                                      \
  } while (false)

class Timer
{
public:
  explicit Timer(const std::string &msg) : m_msg(msg)
  {
    m_start = std::chrono::high_resolution_clock::now();
  }

  ~Timer()
  {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
    std::cout << m_msg << " took " << duration.count() / 1000.0 << "ms" << std::endl;
  }

private:
  std::string m_msg;
  std::chrono::high_resolution_clock::time_point m_start;
};

#define TIMER(msg) Timer timer_##__LINE__(msg);

// read DEBUG flag from environment as a bool
const bool DEBUG = std::getenv("DEBUG") ? std::stoi(std::getenv("DEBUG")) : false;
const bool SILENT = std::getenv("SILENT") ? std::stoi(std::getenv("SILENT")) : false;

void debug(const std::string &msg)
{
  if (DEBUG)
    std::cout << msg << std::endl;
}

void log(const std::string &msg)
{
  if (!SILENT)
    std::cout << msg << std::endl;
}

std::pair<std::vector<uint8_t>, size_t> load_file(const std::string &filename)
{
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file: " + filename);

  size_t file_size = static_cast<size_t>(file.tellg());
  std::vector<uint8_t> buffer(file_size);

  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char *>(buffer.data()), file_size))
    throw std::runtime_error("Failed to read file: " + filename);
  debug("read " + std::to_string(file_size) + " bytes from " + filename);

  return std::make_pair(buffer, file_size);
}

int compress(py::bytes pybytes, const std::string filename)
{
  std::string bytes_str = pybytes;
  size_t input_buffer_len = bytes_str.size();
  std::vector<uint8_t> uncompressed_data(bytes_str.data(), bytes_str.data() + input_buffer_len);
  debug("working with " + std::to_string(input_buffer_len) + " uncompressed bytes");

  uint8_t *device_input_ptrs;
  CUDA_CHECK(cudaMalloc(&device_input_ptrs, input_buffer_len));
  CUDA_CHECK(cudaMemcpy(device_input_ptrs, uncompressed_data.data(), input_buffer_len, cudaMemcpyDefault));

  // start compressing

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  const int chunk_size = 1 << 16;
  nvcompType_t data_type = NVCOMP_TYPE_CHAR;

  GdeflateManager nvcomp_manager{chunk_size, data_type, stream};
  CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);
  uint8_t *comp_buffer;
  CUDA_CHECK(cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));

  // check how long compression takes

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  debug("compression time: " + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()) + "[ms]");

  size_t comp_size = nvcomp_manager.get_compressed_output_size(comp_buffer);
  float comp_ratio = (float)comp_size / (float)input_buffer_len;

  // copy compressed buffer to host memory and then write it to a file

  std::ofstream comp_file(filename, std::ios::binary);
  if (!comp_file.is_open())
    throw std::runtime_error("Failed to open file: " + filename);

  std::vector<uint8_t> comp_buffer_host(comp_size);
  CUDA_CHECK(cudaMemcpy(comp_buffer_host.data(), comp_buffer, comp_size, cudaMemcpyDefault));
  debug("writing compressed buffer to file: " + filename);
  comp_file.write(reinterpret_cast<const char *>(comp_buffer_host.data()), comp_size);
  comp_file.close();

  CUDA_CHECK(cudaFree(comp_buffer));
  CUDA_CHECK(cudaFree(device_input_ptrs));
  CUDA_CHECK(cudaStreamDestroy(stream));

  log("compressed size: " + std::to_string(comp_size) + ", compression ratio: " + std::to_string(comp_ratio));

  return comp_size;
}

torch::ScalarType type_for_name(std::string type_name)
{
  if (type_name == "uint8")
  {
    log("uint8 detected");
    return torch::kUInt8;
  }
  else if (type_name == "int8")
    return torch::kInt8;
  else if (type_name == "int16")
    return torch::kInt16;
  else if (type_name == "int32")
    return torch::kInt32;
  else if (type_name == "int64")
    return torch::kInt64;
  else if (type_name == "float16")
    return torch::kFloat16;
  else if (type_name == "float32")
  {
    log("float32 detected");
    return torch::kFloat32;
  }
  else if (type_name == "float64")
    return torch::kFloat64;
  else
    throw std::runtime_error("Unknown type name: " + type_name);
}

// maybe just decompress from an arbitrary stream + length?
// doing an r2 call with a library probably sucks
// can you just get a fd?
// or a python iterator...?
// maybe you can go into python to grab an http library

// cudaMemcpy 128kb ?
// i5: L1 90K L2 2MB L3 24MB
// 3090: L1 128kb L2 6MB
// check if there's a significant overhead on extra number of copies
// and either 4kb, 128kb or 2-4MB
// sd unet is 1.7 GB, vae 580MB, clip 235MB

std::pair<int, int> decompress(const std::string filename, torch::Tensor tensor)
{
  std::vector<uint8_t> compressed_data;
  size_t input_buffer_len;
  std::tie(compressed_data, input_buffer_len) = load_file(filename);
  uint8_t *comp_buffer;

  std::chrono::steady_clock::time_point copy_begin = std::chrono::steady_clock::now();
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaMalloc(&comp_buffer, input_buffer_len));
  // TODO: use chunked copies
  CUDA_CHECK(cudaMemcpyAsync(comp_buffer, compressed_data.data(), input_buffer_len, cudaMemcpyDefault, stream));
  std::chrono::microseconds copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - copy_begin);

  auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);
  DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
  debug("decompressing into tensor of size " + std::to_string(decomp_config.decomp_data_size));

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  decomp_nvcomp_manager->decompress(static_cast<uint8_t *>(tensor.data_ptr()), comp_buffer, decomp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::chrono::microseconds decomp_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin);
  log("copy time: " + std::to_string(copy_time.count()) + "[µs], decompression time: " + std::to_string(decomp_time.count()) + "[µs]");
  CUDA_CHECK(cudaFree(comp_buffer));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return std::make_pair(copy_time.count(), decomp_time.count());
}

torch::Tensor decompress_new_tensor(const std::string filename, std::vector<int64_t> shape, std::string dtype)
{

  std::vector<uint8_t> compressed_data;
  size_t input_buffer_len;
  std::tie(compressed_data, input_buffer_len) = load_file(filename);
  uint8_t *comp_buffer;

  std::chrono::steady_clock::time_point copy_begin = std::chrono::steady_clock::now();
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaMalloc(&comp_buffer, input_buffer_len));
  // TODO: use chunked copies
  CUDA_CHECK(cudaMemcpyAsync(comp_buffer, compressed_data.data(), input_buffer_len, cudaMemcpyDefault, stream));
  std::chrono::microseconds copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - copy_begin);

  auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);
  DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
  debug("decompressing " + std::to_string(decomp_config.decomp_data_size) + " bytes");

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  auto options = torch::TensorOptions().dtype(type_for_name(dtype)).device(torch::kCUDA);
  torch::Tensor tensor = torch::empty(shape, options);
  log("created tensor");

  decomp_nvcomp_manager->decompress(static_cast<uint8_t *>(tensor.data_ptr()), comp_buffer, decomp_config);
  log("decompressed into tensor of size " + std::to_string(tensor.numel()));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::chrono::microseconds decomp_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin);
  log("copy time: " + std::to_string(copy_time.count()) + "[µs], decompression time: " + std::to_string(decomp_time.count()) + "[µs]");
  CUDA_CHECK(cudaFree(comp_buffer));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return tensor;
}

// intptr_t decompress_lazy(const std::string filename, torch::Tensor tensor)
// {
//   std::vector<uint8_t> compressed_data;
//   size_t input_buffer_len;
//   std::tie(compressed_data, input_buffer_len) = load_file(filename);
//   uint8_t *comp_buffer;

//   std::chrono::steady_clock::time_point copy_begin = std::chrono::steady_clock::now();
//   cudaStream_t stream;
//   CUDA_CHECK(cudaStreamCreate(&stream));
//   CUDA_CHECK(cudaMalloc(&comp_buffer, input_buffer_len));
//   // TODO: use chunked copies
//   CUDA_CHECK(cudaMemcpyAsync(comp_buffer, compressed_data.data(), input_buffer_len, cudaMemcpyDefault, stream));
//   std::chrono::microseconds copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - copy_begin);

//   auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);
//   DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
//   debug("decompressing into tensor of size " + std::to_string(decomp_config.decomp_data_size));

//   std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

//   decomp_nvcomp_manager->decompress(static_cast<uint8_t *>(tensor.data_ptr()), comp_buffer, decomp_config);
//   //CUDA_CHECK(cudaStreamSynchronize(stream));
//   std::chrono::microseconds decomp_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin);
//   log("copy time: " + std::to_string(copy_time.count()) + "[µs], decompression time: " + std::to_string(decomp_time.count()) + "[µs]");
//   CUDA_CHECK(cudaFree(comp_buffer));
//   return reinterpret_cast<intptr_t>(stream);
//   // return stream;
// }

// void finalize_streams(std::vector<intptr_t> streams) {
//   CUDA_CHECK(cudaDeviceSynchronize());
//   for (auto& stream : streams) {
//     cudaStream_t stream_t = reinterpret_cast<cudaStream_t>(stream); // // cast int to cudaStream_t?
//     CUDA_CHECK(cudaStreamDestroy(stream_t));
//   }
// }

// take a list of filenames and tensors and decompress them all
std::vector<std::pair<int, int>> batch_decompress(const std::vector<std::string> filenames, const std::vector<torch::Tensor> tensors)
{
  std::vector<std::pair<int, int>> times;
  for (int i = 0; i < filenames.size(); i++)
  {
    times.push_back(decompress(filenames[i], tensors[i]));
  }
  return times;
}

void batch_decompress_async(const std::vector<std::string> filenames, const std::vector<torch::Tensor> tensors)
{
  std::vector<cudaStream_t> streams;
  int total_copy_time = 0;
  int total_decomp_time = 0;
  // compressed buffers on gpu
  std::vector<uint8_t *> comp_buffers;
  for (int i = 0; i < filenames.size(); i++)
  {
    std::vector<uint8_t> compressed_data;
    size_t input_buffer_len;
    std::tie(compressed_data, input_buffer_len) = load_file(filenames[i]);
    uint8_t *comp_buffer;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    log("created stream " + std::to_string(reinterpret_cast<intptr_t>(stream)));
    streams.push_back(stream);

    std::chrono::steady_clock::time_point copy_begin = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMallocAsync(&comp_buffer, input_buffer_len, stream));
    CUDA_CHECK(cudaMemcpyAsync(comp_buffer, compressed_data.data(), input_buffer_len, cudaMemcpyDefault, stream));
    std::chrono::microseconds copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - copy_begin);
    total_copy_time += copy_time.count();
    comp_buffers.push_back(comp_buffer);
  }
  log("launched all copies");
  for (int i = 0; i < comp_buffers.size(); i++)
  {
    auto decomp_nvcomp_manager = create_manager(comp_buffers[i], streams[i]);
    DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffers[i]);
    debug("decompressing into tensor exptected to have size " + std::to_string(decomp_config.decomp_data_size));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    decomp_nvcomp_manager->decompress(static_cast<uint8_t *>(tensors[i].data_ptr()), comp_buffers[i], decomp_config);
    std::chrono::microseconds decomp_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin);

    // log("copy time: " + std::to_string(copy_time.count()) + "[µs], decompression time: " + std::to_string(decomp_time.count()) + "[µs]");
    log("decompression time: " + std::to_string(decomp_time.count()) + "[µs]");
    CUDA_CHECK(cudaFreeAsync(comp_buffers[i], streams[i]));
    // total_copy_time += copy_time.count();
    total_decomp_time += decomp_time.count();
  }

  log("launched all decompress, waiting for streams to finish");

  std::chrono::steady_clock::time_point sync_begin = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::chrono::microseconds sync_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sync_begin);
  log("cudaDeviceSynchronize took" + std::to_string(sync_time.count()) + "[µs]");

  std::chrono::steady_clock::time_point destroy_begin = std::chrono::steady_clock::now();
  for (int i = 0; i < streams.size(); i++)
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  std::chrono::microseconds destroy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - destroy_begin);
  log("destroyed " + std::to_string(streams.size()) + " streams, took " + std::to_string(destroy_time.count()) + "[µs]");
  log("total copy time: " + std::to_string(total_copy_time) + "[µs], total decomp time: " + std::to_string(total_decomp_time) + "[µs]");
}

// why is batch_decompress_async slower than batch_decompress?

std::vector<torch::Tensor> good_batch_decompress_threaded(
    const std::vector<std::string> &filenames,
    const std::vector<std::vector<int64_t>> &tensor_shapes,
    const std::vector<std::string> &dtypes)
{
  int total_copy_time = 0;
  int total_decomp_time = 0;
  std::vector<std::future<void>> futures;
  std::vector<torch::Tensor> tensors = std::vector<torch::Tensor>(filenames.size());

  std::vector<cudaStream_t> streams((int)filenames.size());
  std::vector<nvcompStatus_t *> statuses((int)filenames.size());

  for (int i = 0; i < filenames.size(); i++)
  {

    auto options = torch::TensorOptions().dtype(type_for_name(dtypes[i])).device(torch::kCUDA);
    tensors[i] = torch::empty(tensor_shapes[i], options);
    auto empty_checksum = tensors[i].sum().item<int64_t>();

    futures.emplace_back(std::async(std::launch::async, [&, i, empty_checksum]()
                                    {
                                      std::vector<uint8_t> compressed_data;
                                      size_t input_buffer_len;
                                      std::tie(compressed_data, input_buffer_len) = load_file(filenames[i]);

                                      uint8_t *comp_buffer;

                                      CUDA_CHECK(cudaStreamCreate(&streams[i]));
                                      cudaStream_t stream = streams[i % streams.size()];

                                      log("created stream");
                                      CUDA_CHECK(cudaMallocAsync(&comp_buffer, input_buffer_len, stream));
                                      log("malloced buffer");
                                      auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);
                                      log("create_manager");
                                      DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
                                      log("configured");
                                      // auto options = torch::TensorOptions().dtype( type_for_name(dtypes[i])).device(torch::kCUDA);
                                      // tensors[i] = torch::empty(tensor_shapes[i], options);

                                      std::chrono::steady_clock::time_point copy_begin = std::chrono::steady_clock::now();
                                      CUDA_CHECK(cudaMemcpyAsync(comp_buffer, compressed_data.data(), input_buffer_len, cudaMemcpyDefault, stream));
                                      std::chrono::microseconds copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - copy_begin);
                                      log("memcopy time: " + std::to_string(copy_time.count()) + "[µs]");
                                      std::chrono::steady_clock::time_point decomp_begin = std::chrono::steady_clock::now();

                                      // auto empty_checksum = tensors[i].sum().item<int64_t>();

                                      log("empty tensor checksum: " + std::to_string(empty_checksum));
                                      decomp_nvcomp_manager->decompress(static_cast<uint8_t *>(tensors[i].data_ptr()), comp_buffer, decomp_config);

                                      std::chrono::microseconds decomp_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - decomp_begin);
                                      log("decomp time: " + std::to_string(decomp_time.count()) + "[µs]");

                                      CUDA_CHECK(cudaFreeAsync(comp_buffer, stream));
                                      // get checksum for decompressed tensor
                                      auto checksum = tensors[i].sum().item<int64_t>();
                                      log("decompressed tensor with checksum: " + std::to_string(checksum) + ", should be different compared to empty tensor checksum: " + std::to_string(empty_checksum));

                                      // log("copy time: " + std::to_string(copy_time.count()) + "[µs], decompression time: " + std::to_string(decomp_time.count()) + "[µs]");
                                      total_copy_time += copy_time.count();
                                      total_decomp_time += decomp_time.count();
                                      //
                                      statuses[i] = decomp_config.get_status();
                                      log("decompression status: " + std::to_string(*statuses[i]));
                                      // configs.emplace_back(decomp_config);
                                      // print decompression status
                                    }));
    log("made future");
  }

  log("launched all async operations, waiting for them to finish");
  auto wait_begin = std::chrono::steady_clock::now();
  for (auto &f : futures)
  {
    f.wait();
  }
  std::chrono::microseconds wait_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wait_begin);
  log("waited for all futures to finish, took " + std::to_string(wait_time.count()) + "[µs]");
  auto cuda_sync_begin = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::chrono::microseconds cuda_sync_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - cuda_sync_begin);
  log("cudaDeviceSynchronize took " + std::to_string(cuda_sync_time.count()) + "[µs]");
  for (auto &s : streams)
  {
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaStreamDestroy(s));
  }
  log("total copy time: " + std::to_string(total_copy_time) + "[µs], total decomp time: " + std::to_string(total_decomp_time) + "[µs]");
  CUDA_CHECK(cudaDeviceSynchronize());
  return tensors;
}

std::vector<torch::Tensor> good_batch_decompress_threadpool(
    const std::vector<std::string> &filenames,
    const std::vector<std::vector<int64_t>> &tensor_shapes,
    const std::vector<std::string> &dtypes)
{
  int total_copy_time = 0;
  int total_decomp_time = 0;
  std::vector<std::future<void>> futures;
  std::vector<torch::Tensor> tensors = std::vector<torch::Tensor>(filenames.size());

  std::vector<cudaStream_t> streams((int)filenames.size());
  std::vector<nvcompStatus_t *> statuses((int)filenames.size());

  // decide which threads should handle which indexes
  int threads = std::min((uint)filenames.size(), std::thread::hardware_concurrency());
  std::vector<std::vector<int>> thread_to_indexes(threads);
  for (int i = 0; i < filenames.size(); i++)
  {
    thread_to_indexes[i % threads].push_back(i);
  }

  for (auto indexes : thread_to_indexes)
  {
    futures.emplace_back(std::async(std::launch::async, [&, indexes]()
                                    {
      for (int i : indexes) {  
        std::vector<uint8_t> compressed_data;
        size_t input_buffer_len;
        std::tie(compressed_data, input_buffer_len) = load_file(filenames[i]);

        uint8_t* comp_buffer;

        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        cudaStream_t stream = streams[i % streams.size()];      

        log("created stream");
        CUDA_CHECK(cudaMallocAsync(&comp_buffer, input_buffer_len, stream));
        log("malloced buffer");
        auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);
        log("create_manager");
        DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
        log("configured");
        // auto options = torch::TensorOptions().dtype( type_for_name(dtypes[i])).device(torch::kCUDA);
        // tensors[i] = torch::empty(tensor_shapes[i], options);


        std::chrono::steady_clock::time_point copy_begin = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(comp_buffer, compressed_data.data(), input_buffer_len, cudaMemcpyDefault, stream));
        std::chrono::microseconds copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - copy_begin);
        log("memcopy time: " + std::to_string(copy_time.count()) + "[µs]");
        std::chrono::steady_clock::time_point decomp_begin = std::chrono::steady_clock::now();


        auto options = torch::TensorOptions().dtype( type_for_name(dtypes[i])).device(torch::kCUDA);
        tensors[i] = torch::empty(tensor_shapes[i], options);
        auto empty_checksum = tensors[i].sum().item<int64_t>();

        log("empty tensor checksum: " + std::to_string(empty_checksum));
        decomp_nvcomp_manager->decompress(static_cast<uint8_t*>(tensors[i].data_ptr()), comp_buffer, decomp_config);
        
        std::chrono::microseconds decomp_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - decomp_begin);
        log("decomp time: " + std::to_string(decomp_time.count()) + "[µs]");

        CUDA_CHECK(cudaFreeAsync(comp_buffer, stream));
        // get checksum for decompressed tensor
        auto checksum = tensors[i].sum().item<int64_t>();
        log("decompressed tensor with checksum: " + std::to_string(checksum) + ", should be different compared to empty tensor checksum: " + std::to_string(empty_checksum));

        // log("copy time: " + std::to_string(copy_time.count()) + "[µs], decompression time: " + std::to_string(decomp_time.count()) + "[µs]");
        total_copy_time += copy_time.count();
        total_decomp_time += decomp_time.count();
        // 
        statuses[i] = decomp_config.get_status();
        log("decompression status: " + std::to_string(*statuses[i]));
        // configs.emplace_back(decomp_config);
        // print decompression status
      }
      log("thread done"); }));
    log("made future");
  }

  log("launched all async operations, waiting for them to finish");
  auto wait_begin = std::chrono::steady_clock::now();
  for (auto &f : futures)
  {
    f.wait();
  }
  std::chrono::microseconds wait_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - wait_begin);
  log("waited for all futures to finish, took " + std::to_string(wait_time.count()) + "[µs]");
  auto cuda_sync_begin = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::chrono::microseconds cuda_sync_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - cuda_sync_begin);
  log("cudaDeviceSynchronize took " + std::to_string(cuda_sync_time.count()) + "[µs]");
  for (auto &s : streams)
  {
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaStreamDestroy(s));
  }
  log("total copy time: " + std::to_string(total_copy_time) + "[µs], total decomp time: " + std::to_string(total_decomp_time) + "[µs]");
  CUDA_CHECK(cudaDeviceSynchronize());
  return tensors;
}

// void batch_decompress_threaded(const std::vector<std::string>& filenames, const std::vector<torch::Tensor>& tensors) {
//   int total_copy_time = 0;
//   int total_decomp_time = 0;
//   std::vector<std::future<void>> futures;
//   std::vector<cudaStream_t> streams;
//   for (int i = 0; i < filenames.size(); i++) {
//       std::vector<uint8_t> compressed_data;
//       size_t input_buffer_len;
//       std::tie(compressed_data, input_buffer_len) = load_file(filenames[i]);

//       futures.push_back(std::async(std::launch::async, [&, i, compressed_data = std::move(compressed_data), input_buffer_len]() {
//           uint8_t* comp_buffer;

//           cudaStream_t stream;
//           log("stream");
//           CUDA_CHECK(cudaStreamCreate(&stream));
//           log("created stream");

//           CUDA_CHECK(cudaMallocAsync(&comp_buffer, input_buffer_len, stream));
//           log("malloced buffer");
//           auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);
//           log("create_manager");
//           DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
//           log("configured");

//           std::chrono::steady_clock::time_point copy_begin = std::chrono::steady_clock::now();
//           CUDA_CHECK(cudaMemcpyAsync(comp_buffer, compressed_data.data(), input_buffer_len, cudaMemcpyDefault, stream));
//           log("memcpy");
//           std::chrono::microseconds copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - copy_begin);

//           std::chrono::steady_clock::time_point decomp_begin = std::chrono::steady_clock::now();
//           decomp_nvcomp_manager->decompress(static_cast<uint8_t*>(tensors[i].data_ptr()), comp_buffer, decomp_config);
//           log("decomp");
//           std::chrono::microseconds decomp_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - decomp_begin);

//           CUDA_CHECK(cudaFree(comp_buffer));
//           streams.push_back(stream);
//           // CUDA_CHECK(cudaStreamDestroy(stream));

//           log("copy time: " + std::to_string(copy_time.count()) + "[µs], decompression time: " + std::to_string(decomp_time.count()) + "[µs]");
//           total_copy_time += copy_time.count();
//           total_decomp_time += decomp_time.count();
//       }));
//       log("made future");
//   }

//   log("launched all async operations, waiting for them to finish");
//   for (auto& f : futures) {
//       f.wait();
//   }
//   CUDA_CHECK(cudaDeviceSynchronize());
//   for (auto& s : streams) {
//       CUDA_CHECK(cudaStreamDestroy(s));
//   }
//   log("total copy time: " + std::to_string(total_copy_time) + "[µs], total decomp time: " + std::to_string(total_decomp_time) + "[µs]");
// }

// std::shared_ptr<torch::Tensor> make_tensor() {
//     auto tensor = std::make_shared<torch::Tensor>(torch::ones(5));
//     std::cout << "made tensor" << std::endl;
//     return tensor;
// }

torch::Tensor make_tensor(torch::Tensor foo)
{
  auto tensor_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, 0);
  torch::Tensor tensor = torch::empty({100}, tensor_options);
  // foo.data = tensor
  return tensor;
}

// torch::Tensor d_sigmoid(torch::Tensor z) {
//   auto s = torch::sigmoid(z);
//   return (1 - s) * s;
// }

PYBIND11_MODULE(_nyacomp, m)
{

  m.doc() = R"pbdoc(python bindings for nvcomp with torch)pbdoc";

  m.def("compress", &compress, R"pbdoc(
        compress
    )pbdoc",
        py::arg("data"), py::arg("filename"));

  m.def("decompress", &decompress, R"pbdoc(
        decompress
    )pbdoc",
        py::arg("filename"), py::arg("dest_tensor"));

  // m.def("decompress_lazy", &decompress_lazy, "start decompressing but don't sync", py::arg("filename"), py::arg("dest_tensor"));

  m.def("decompress_new_tensor", &decompress_new_tensor, "decompress to a new tensor", py::arg("filename"), py::arg("shape"), py::arg("dtype"));

  // m.def("foo", &foo, "starts work and returns a stream");
  // m.def("finalize_streams", &finalize_streams, "finalize and destroy a list of streams", py::arg("streams"));

  m.def("decompress_batch", &batch_decompress, "decompress batch", py::arg("filenames"), py::arg("dest_tensors"));
  m.def("decompress_batch_async", &batch_decompress_async, "async decompress batch", py::arg("filenames"), py::arg("dest_tensors"));
  // m.def("decompress_batch_threaded", &batch_decompress_threaded, "threaded decompress batch", py::arg("filenames"), py::arg("dest_tensors"));
  m.def("good_batch_decompress_threaded", &good_batch_decompress_threaded, "good decompress batch", py::arg("filenames"), py::arg("shapes"), py::arg("dtypes"));
  m.def("good_batch_decompress_threadpool", &good_batch_decompress_threadpool, "good decompress batch", py::arg("filenames"), py::arg("shapes"), py::arg("dtypes"));

  // m.def("sigmoid", &d_sigmoid, "sigmod fn");

  m.def("make_tensor", &make_tensor, py::return_value_policy::take_ownership);

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
