#include <pybind11/pybind11.h>
#include "nvcomp/gdeflate.hpp"
// #include "nvcomp/cascaded.hpp"
// #include "nvcomp/bitcomp.hpp"
// #include "nvcomp/lz4.hpp"
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
      std::cerr << "[CUDA_CHECK] Cuda error: " << cudaGetErrorString(err) << std::endl; \
      std::cerr << "code: " << #cond << std::endl;                         \
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

int getenv(const char *name, int default_value) {
  return std::getenv(name) ? std::stoi(std::getenv(name)) : default_value;
}

// cast to bool
// const bool DEBUG = static_cast<bool>(getenv("DEBUG", 0));
const bool DEBUG = getenv("DEBUG", 0); 
const bool SILENT = getenv("SILENT", 0);

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
  // allocate pinned cudaHost memory for the file?
  // CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void **>(&file), file_size, cudaHostAllocDefault));
  std::vector<uint8_t> buffer(file_size);

  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char *>(buffer.data()), file_size))
    throw std::runtime_error("Failed to read file: " + filename);
  debug("read " + std::to_string(file_size) + " bytes from " + filename);

  return std::make_pair(buffer, file_size);
}

// std::pair<uint8_t*, size_t> load_file_pinned(const std::string &filename) {
//   std::ifstream file(filename, std::ios::binary | std::ios::ate);
//   if (!file.is_open())
//     throw std::runtime_error("Failed to open file: " + filename);

//   size_t file_size = static_cast<size_t>(file.tellg());
//   uint8_t *buffer;
//   CUDA_CHECK(cudaMallocHost(&buffer, file_size));

//   file.seekg(0, std::ios::beg);
//   if (!file.read(reinterpret_cast<char *>(buffer), file_size))
//     throw std::runtime_error("Failed to read file: " + filename);
//   debug("read " + std::to_string(file_size) + " bytes from " + filename);

//   return std::make_pair(buffer, file_size);
// }

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
  // nvcompBatchedGdeflateOpts_t opts = {1}; // :( :(

  // LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
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
    debug("uint8 detected");
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
    debug("float32 detected");
    return torch::kFloat32;
  }
  else if (type_name == "float64")
    return torch::kFloat64;
  else
    throw std::runtime_error("Unknown type name: " + type_name);
}

torch::Tensor make_tensor(const std::vector<int64_t> &shape, const std::string &dtype)
{
  auto options = torch::TensorOptions().dtype(type_for_name(dtype)).device(torch::kCUDA);
  return torch::empty(shape, options);
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

// take a list of filenames and tensors and decompress them all
std::vector<std::pair<int, int>> batch_decompress(const std::vector<std::string> filenames, const std::vector<torch::Tensor> tensors)
{
  std::vector<std::pair<int, int>> times;
  for (int i = 0; i < filenames.size(); i++)
    times.push_back(decompress(filenames[i], tensors[i]));
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

std::vector<torch::Tensor> batch_decompress_async_new(const std::vector<std::string> &filenames, const std::vector<std::vector<int64_t>> &shapes, const std::vector<std::string> &dtypes)
{
  std::chrono::steady_clock::time_point first_begin = std::chrono::steady_clock::now();
  std::vector<cudaStream_t> streams;
  int total_copy_time = 0;
  int total_decomp_time = 0;
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
    comp_buffers.emplace_back(comp_buffer);
  }
  log("launched all copies");
  std::vector<torch::Tensor> tensors(filenames.size());

  for (int i = 0; i < comp_buffers.size(); i++)
  {
    auto options = torch::TensorOptions().dtype(type_for_name(dtypes[i])).device(torch::kCUDA);
    tensors[i] = torch::empty(shapes[i], options);
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
  auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - first_begin).count();
  log("total copy time: " + std::to_string(total_copy_time) + "[µs], total decomp time: " + std::to_string(total_decomp_time) + "[µs], overall time: " + std::to_string(total_time) + "[ms]");
  return tensors;
}



std::vector<torch::Tensor> batch_decompress_threadpool(
    const std::vector<std::string> &filenames,
    const std::vector<std::vector<int64_t>> &tensor_shapes,
    const std::vector<std::string> &dtypes)
{
  auto start_time = std::chrono::steady_clock::now();

  if (filenames.size() != tensor_shapes.size() || filenames.size() != dtypes.size())
    throw std::invalid_argument("All input vectors should have the same size");

  int num_files = static_cast<int>(filenames.size());
  int num_threads = std::min(num_files, getenv("NUM_THREADS", static_cast<int>(std::thread::hardware_concurrency())));


  std::vector<std::future<std::pair<int64_t, int64_t>>> futures;
  std::vector<torch::Tensor> tensors(num_files);
  std::atomic<int> total_copy_time(0), total_decomp_time(0);

  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
  // ampere: 128 concurrent kernels ... (3090, A30, etc are 8.6)
  // v100 is 7.0, but still 128
  // std::vector<cudaStream_t> streams(num_threads); 
  int num_streams = getenv("NUM_STREAMS", 128);
  int streams_per_thread = num_streams / num_threads; 
  std::vector<std::vector<cudaStream_t>> streams = std::vector<std::vector<cudaStream_t>>(num_threads, std::vector<cudaStream_t>(streams_per_thread));
  
  log("Using " + std::to_string(num_threads) + " threads and " + std::to_string(streams_per_thread) + " streams per thread for " + std::to_string(num_files) + " files");

  // initialize the primary context 
  CUDA_CHECK(cudaSetDevice(0));

  // auto create_stream_begin = std::chrono::steady_clock::now();
  // for (auto &stream : streams) 
  //   CUDA_CHECK(cudaStreamCreate(&stream));
  // log("creating streams took " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - create_stream_begin).count()) + "[µs]");
  
  std::vector<std::vector<int>> thread_to_indexes(num_threads);
  for (int i = 0; i < num_files; i++)
    thread_to_indexes[i % num_threads].push_back(i);


  for (int thread_id = 0; thread_id < num_threads; thread_id++) {
    auto indexes = thread_to_indexes[thread_id];
    
    futures.emplace_back(std::async(std::launch::async, [indexes, thread_id, &streams, &tensors, &filenames, &tensor_shapes, &dtypes, &streams_per_thread]() {
      log("started thread " + std::to_string(thread_id));
      CUDA_CHECK(cudaSetDevice(0)); // this ... sets the context?
      
      int64_t thread_copy_time = 0, thread_decomp_time = 0;
      auto create_stream_begin = std::chrono::steady_clock::now();
      for (int stream_id = 0; stream_id < streams_per_thread; stream_id++) 
        CUDA_CHECK(cudaStreamCreate(&streams[thread_id][stream_id]));
      

      log(std::to_string(thread_id) + ": creating streams took " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - create_stream_begin).count()) + "[µs]");
      // cudaStream_t stream = streams[thread_id];

      
      // for (int i : indexes) {
      for (auto job_number = 0; job_number < (int)indexes.size(); job_number++) {
        int i = indexes[job_number];
        auto prefix = "thread " + std::to_string(thread_id) + ", tensor " + std::to_string(i) + " (job " + std::to_string(job_number) + "): ";
        auto file_start_time = std::chrono::steady_clock::now();

        cudaStream_t stream = streams[thread_id][job_number % streams_per_thread];
        // convert stream to int for logging
        int stream_int = static_cast<int>(reinterpret_cast<intptr_t>(stream));
        

        std::vector<uint8_t> compressed_data;
        size_t input_buffer_len;
        std::tie(compressed_data, input_buffer_len) = load_file(filenames[i]);

        // uint8_t* host_compressed_data;
        // size_t input_buffer_len;
        // std::tie(host_compressed_data, input_buffer_len) = load_file_pinned(filenames[i]);

        uint8_t* comp_buffer;
        debug(prefix + "allocating device memory with stream " + std::to_string(stream_int));
        CUDA_CHECK(cudaMallocAsync(&comp_buffer, input_buffer_len, stream));

        
        auto copy_begin = std::chrono::steady_clock::now();
        debug(prefix + "copying to device with stream " + std::to_string(stream_int));
        // CUDA_CHECK(cudaMemcpyAsync(comp_buffer, host_compressed_data, input_buffer_len, cudaMemcpyDefault, stream));
        CUDA_CHECK(cudaMemcpyAsync(comp_buffer, compressed_data.data(), input_buffer_len, cudaMemcpyDefault, stream));
        std::chrono::microseconds copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - copy_begin);

        // CUDA_CHECK(cudaFreeHost(host_compressed_data));

        tensors[i] = make_tensor(tensor_shapes[i], dtypes[i]);
        debug(prefix + "creating manager with stream " + std::to_string(stream_int));
        // this syncs the stream
        auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);
        debug(prefix + "configuring decomp");
        DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
        auto decomp_begin = std::chrono::steady_clock::now();
        debug(prefix + "decompressing");

        try {
          decomp_nvcomp_manager->decompress(static_cast<uint8_t*>(tensors[i].data_ptr()), comp_buffer, decomp_config);
        } catch (const std::exception& e) {
          // check each of the resource handles to see if they're valid
          log(prefix + "exception: " + std::string(e.what()));
          // check cuda status
          cudaError_t err = cudaGetLastError();
          log(prefix + "cuda error: " + std::string(cudaGetErrorString(err)));
          // check stream status
          cudaError_t err2 = cudaStreamQuery(stream);
          log(prefix + "cuda stream error: " + std::string(cudaGetErrorString(err2)));
          log(prefix + "throwing exception");
          throw e;
        }
        // decomp_nvcomp_manager->decompress(static_cast<uint8_t*>(tensors[i].data_ptr()), comp_buffer, decomp_config);
        std::chrono::microseconds decomp_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - decomp_begin);
        
        log(prefix + "decompressed " + std::to_string(i) + " in " + std::to_string(decomp_time.count()) + "[µs], freeing with stream " + std::to_string(stream_int) + "");
        CUDA_CHECK(cudaFreeAsync(comp_buffer, stream));

        auto file_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - file_start_time).count();
        // total_copy_time += copy_time.count();
        // total_decomp_time += decomp_time.count();
        log(prefix + "processed in " + std::to_string(file_elapsed_time) + " ms"); 
        thread_copy_time += copy_time.count();
        thread_decomp_time += decomp_time.count();
      }
      // auto sync_start = std::chrono::steady_clock::now();
      // CUDA_CHECK(cudaStreamSynchronize(stream));
      // CUDA_CHECK(cudaStreamDestroy(stream));
      // log(prefix + "sync time: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - sync_start).count()) + "[µs]");
      return std::make_pair(thread_copy_time, thread_decomp_time);
    }));
  }
  log("waiting for futures");

  // for (auto &f : futures){
  for (int i = 0; i < (int)futures.size(); i++) {
    int copy, decomp;
    debug("waiting for future " + std::to_string(i));
    std::tie(copy, decomp) = futures[i].get();
    log("got future " + std::to_string(i));
    total_copy_time += copy;
    total_decomp_time += decomp;
  }
  for (auto &thread_streams : streams) 
    for (auto &stream : thread_streams)
      CUDA_CHECK(cudaStreamSynchronize(stream));
  for (auto &thread_streams : streams) 
    for (auto &stream : thread_streams)
      CUDA_CHECK(cudaStreamDestroy(stream));
  

  auto device_sync_start = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaDeviceSynchronize());
  log("Device sync time: " + std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - device_sync_start).count()) + "[µs]");

  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  log("Total processing time: " + std::to_string(elapsed_time) + " ms for " + std::to_string(num_files) + " tensors");
  log("Total copy time: " + std::to_string(total_copy_time) + "[µs], total decomp time: " + std::to_string(total_decomp_time) + "[µs]");

  return tensors;
}


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

  m.def("decompress_new_tensor", &decompress_new_tensor, "decompress to a new tensor", py::arg("filename"), py::arg("shape"), py::arg("dtype"));

  m.def("decompress_batch", &batch_decompress, "decompress batch", py::arg("filenames"), py::arg("dest_tensors"));
  m.def("decompress_batch_async", &batch_decompress_async, "async decompress batch", py::arg("filenames"), py::arg("dest_tensors"));
  m.def("decompress_batch_async_new", &batch_decompress_async_new, "decomp", py::arg("filenames"), py::arg("shapes"), py::arg("dtypes"));

  m.def("batch_decompress_threadpool", &batch_decompress_threadpool, "good decompress batch (limit)", py::arg("filenames"), py::arg("shapes"), py::arg("dtypes"));


  // m.def("lowlevel_example", &lowlevel_example, "lowlevel_example", py::arg("data"), py::arg("length"));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
