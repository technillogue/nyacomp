#include <assert.h>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <iostream>
#include <spawn.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "nvcomp/gdeflate.hpp"
#include "nvcomp.hpp"
// #include "nvcomp/lz4.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"
// #include "nvToolsExt.h"


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace nvcomp;
namespace py = pybind11;

#define CUDA_CHECK(cond)                                                   \
  do {                                                                     \
    cudaError_t err = cond;                                                \
    if (err != cudaSuccess)                                                \
    {                                                                      \
      std::cerr << "[CUDA_CHECK] Cuda error: " << cudaGetErrorString(err) << std::endl; \
      std::cerr << "code: " << #cond << std::endl;                         \
      exit(1);                                                             \
    }                                                                      \
  } while (false)

char* getenv_str(const char* name, const char* default_value) {
  auto value = std::getenv(name);
  return (value == nullptr || value[0] == '\0') ? const_cast<char*>(default_value) : value;
}}

int getenv(const char* name, int default_value) {
  auto value = std::getenv(name);
  return (value == nullptr || value[0] == '\0') ? default_value : std::stoi(value);
}

const bool DEBUG = getenv("DEBUG", 0);
const bool SILENT = getenv("SILENT", 0);
const bool DEBUG_MALLOC = getenv("DEBUG_MALLOC", 0);

void debug(const std::string& msg) {
  if (DEBUG)
    std::cout << msg << std::endl;
}

void log(const std::string& msg) {
  if (!SILENT)
    std::cout << msg << std::endl;
}

void debug_malloc(const std::string& msg) {
  if (DEBUG_MALLOC)
    std::cout << msg << std::endl;
}
// size_t round_up_kb(size_t size) { return (size + 1023) & -1024; }

std::string pprint(cudaStream_t stream) {
  std::stringstream ss;
  ss << std::hex << stream;
  return ss.str();
}

std::string pprint(std::chrono::duration<int64_t, std::nano> duration) {
  if (duration < std::chrono::microseconds(1000))
    return std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(duration).count()) + "µs";
  if (duration < std::chrono::milliseconds(2000)) // at least 2ms feels like better precision
    return std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0) + "ms";
  return std::to_string(std::chrono::duration_cast<std::chrono::seconds>(duration).count()) + "s";

}

std::string pprint(size_t bytes, int precision = 2) {
  std::stringstream ss;
  const char* units[] = { "B", "KB", "MB", "GB" };
  double value = bytes;
  int unit = 0;
  while (value >= 1024 && unit < 4) {
    value /= 1024;
    unit++;
  }
  if (std::fmod(value, 1) == 0)
    ss << std::fixed << std::setprecision(0);
  else
    ss << std::fixed << std::setprecision(precision);
  ss << value << " " << units[unit];
  return ss.str();
}

std::string pprint_throughput(size_t bytes, std::chrono::duration<int64_t, std::nano> duration) {
  return pprint(bytes * 1e9 / std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count()) + "/s";
}


std::pair<std::vector<uint8_t>, size_t> load_file(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file: " + filename);

  size_t file_size = static_cast<size_t>(file.tellg());
  std::vector<uint8_t> buffer(file_size);

  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size))
    throw std::runtime_error("Failed to read file: " + filename);
  debug("read " + pprint(file_size) + " bytes from " + filename);

  return std::make_pair(buffer, file_size);
}

std::pair<uint8_t*, size_t> load_file_wrapper(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file: " + filename);

  size_t file_size = static_cast<size_t>(file.tellg());
  uint8_t* buffer = new uint8_t[file_size];

  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char*>(buffer), file_size))
    throw std::runtime_error("Failed to read file: " + filename);
  debug("read " + pprint(file_size) + " bytes from " + filename);

  return std::make_pair(buffer, file_size);
}


size_t get_fsize(std::string filename) {
  struct stat st;
  if (stat(filename.c_str(), &st) != 0)
    return (long)0;
  return st.st_size;
}

size_t get_fsize(int fd) {
  struct stat st;
  if (fstat(fd, &st) != 0)
    return (long)0;
  return st.st_size;
}


int compress(py::bytes pybytes, const std::string filename) {
  std::string bytes_str = pybytes;
  size_t input_buffer_len = bytes_str.size();
  std::vector<uint8_t> uncompressed_data(bytes_str.data(), bytes_str.data() + input_buffer_len);
  debug("working with " + pprint(input_buffer_len) + " uncompressed bytes");

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint8_t* device_input_ptrs;
  CUDA_CHECK(cudaMallocAsync(&device_input_ptrs, input_buffer_len, stream));
  CUDA_CHECK(cudaMemcpyAsync(device_input_ptrs, uncompressed_data.data(), input_buffer_len, cudaMemcpyDefault, stream));


  const int chunk_size = 1 << 16;
  // nvcompType_t data_type = NVCOMP_TYPE_CHAR;
  // LZ4Manager nvcomp_manager{chunk_size, data_type, stream};

  // 0 : high-throughput, low compression ratio (default) // only supported, lolsob
  // 1 : low-throughput, high compression ratio
  // 2 : highest-throughput, entropy-only compression (use for symmetric compression/decompression performance

  GdeflateManager nvcomp_manager{ chunk_size, nvcompBatchedGdeflateDefaultOpts, stream, NoComputeNoVerify };

  CompressionConfig comp_config = nvcomp_manager.configure_compression(input_buffer_len);
  uint8_t* comp_buffer;
  CUDA_CHECK(cudaMallocAsync(&comp_buffer, comp_config.max_compressed_buffer_size, stream));

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  nvcomp_manager.compress(device_input_ptrs, comp_buffer, comp_config);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  debug("compression time: " + pprint(end - begin));

  size_t comp_size = nvcomp_manager.get_compressed_output_size(comp_buffer);
  float comp_ratio = (float)comp_size / (float)input_buffer_len;

  // copy compressed buffer to host memory and then write it to a file

  std::ofstream comp_file(filename, std::ios::binary);
  if (!comp_file.is_open())
    throw std::runtime_error("Failed to open file: " + filename);

  std::vector<uint8_t> comp_buffer_host(comp_size);
  CUDA_CHECK(cudaMemcpyAsync(comp_buffer_host.data(), comp_buffer, comp_size, cudaMemcpyDefault, stream));
  debug("writing compressed buffer to file: " + filename);
  comp_file.write(reinterpret_cast<const char*>(comp_buffer_host.data()), comp_size);
  comp_file.close();

  CUDA_CHECK(cudaFreeAsync(comp_buffer, stream));
  CUDA_CHECK(cudaFreeAsync(device_input_ptrs, stream));
  CUDA_CHECK(cudaStreamDestroy(stream));

  log("compressed size: " + pprint(comp_size) + ", compression ratio: " + std::to_string(comp_ratio));

  return comp_size;
}

torch::ScalarType type_for_name(std::string type_name) {
  if (type_name == "uint8")
    return torch::kUInt8;
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
    return torch::kFloat32;
  else if (type_name == "float64")
    return torch::kFloat64;
  else
    throw std::runtime_error("Unknown type name: " + type_name);
}

torch::Tensor make_tensor(const std::vector<int64_t>& shape, const std::string& dtype) {
  auto options = torch::TensorOptions().dtype(type_for_name(dtype)).device(torch::kCUDA).memory_format(torch::MemoryFormat::Contiguous);
  return torch::empty(shape, options);
}


torch::Tensor decompress(const std::string filename, std::vector<int64_t> shape, std::string dtype) {
  std::vector<uint8_t> compressed_data;
  size_t input_buffer_len;
  std::tie(compressed_data, input_buffer_len) = load_file(filename);
  uint8_t* comp_buffer;

  std::chrono::steady_clock::time_point copy_begin = std::chrono::steady_clock::now();
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaMalloc(&comp_buffer, input_buffer_len));
  CUDA_CHECK(cudaMemcpyAsync(comp_buffer, compressed_data.data(), input_buffer_len, cudaMemcpyDefault, stream));
  std::chrono::microseconds copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - copy_begin);

  auto decomp_nvcomp_manager = create_manager(comp_buffer, stream);
  DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
  debug("decompressing " + std::to_string(decomp_config.decomp_data_size) + " bytes");

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  auto options = torch::TensorOptions().dtype(type_for_name(dtype)).device(torch::kCUDA);
  torch::Tensor tensor = torch::empty(shape, options);
  log("created tensor");

  decomp_nvcomp_manager->decompress(static_cast<uint8_t*>(tensor.data_ptr()), comp_buffer, decomp_config);
  log("decompressed into tensor of size " + std::to_string(tensor.numel()));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::chrono::microseconds decomp_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin);
  log("copy time: " + std::to_string(copy_time.count()) + "[µs], decompression time: " + std::to_string(decomp_time.count()) + "[µs]");
  CUDA_CHECK(cudaFree(comp_buffer));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return tensor;
}


class FileLoader {
public:
  FileLoader(size_t total_size, size_t num_threads)
    : shared_buffer(nullptr), thread_offsets(num_threads, 0), thread_sizes(num_threads, 0), global_offset(0),
    total_size(total_size), is_ready(false) /*, remaining_releases(num_threads) */ {
    // launch a single thread to allocate the buffer
    alloc_future = std::async(std::launch::async, [this, total_size]() {
      auto start = std::chrono::steady_clock::now();
      debug_malloc("alocating shared " + pprint(total_size) + " host buffer");
      CUDA_CHECK(cudaMallocHost(&shared_buffer, total_size));
      log("allocated shared " + pprint(total_size) + " in " + pprint(std::chrono::steady_clock::now() - start));
      is_ready.store(true, std::memory_order_release);
      });
  }

  ~FileLoader() {
    if (shared_buffer == nullptr)
      return;
    size_t used = global_offset.load();
    auto warning = (total_size - used) > 0 ? ("only " + pprint(used) + " used, but " + pprint(total_size - used) + " was unused. ") : "";
    auto start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaFreeHost(shared_buffer));
    log(warning + "freed " + pprint(total_size) + " FileLoader buffer in " + pprint(std::chrono::steady_clock::now() - start));
  }

  // std::atomic<size_t> remaining_releases;
  // static void release(void* arg) {
  //   FileLoader* loader = static_cast<FileLoader*>(arg);
  //   auto remaining = loader->remaining_releases.fetch_sub(1);
  //   if (remaining == 1) {
  //     log("all threads have released file loader, delete this");
  //     delete loader;
  //   } else log("remaining FileLoader releases: " + std::to_string(remaining - 1));
  // }

  uint8_t* get_buffer(size_t size, size_t thread_id) {
    if (size < 1)
      throw std::runtime_error("Requested size " + pprint(size) + " is too small");
    if (is_ready.load(std::memory_order_acquire) == false)
      alloc_future.wait();
    if (thread_sizes[thread_id] == 0 && thread_offsets[thread_id] == 0) {
      size_t offset = global_offset.fetch_add(size, std::memory_order_relaxed);
      if (offset + size > total_size) {
        size_t remaining = total_size - offset;
        throw std::runtime_error("Shared buffer is not large enough to accommodate " + pprint(size) + " bytes requested by thread " + std::to_string(thread_id) + ", " + pprint(remaining) + "  total bytes");
      }
      thread_offsets[thread_id] = offset;
      thread_sizes[thread_id] = size;
    }
    else {
      if (size > thread_sizes[thread_id]) {
        log("Requested size " + pprint(size) + " is larger than than " + pprint(thread_sizes[thread_id]) + " previously allocated for thread " + std::to_string(thread_id));
        throw std::runtime_error("Requested size " + pprint(size) + " is larger than than " + pprint(thread_sizes[thread_id]) + " previously allocated for thread " + std::to_string(thread_id));
      }
    }
    return shared_buffer + thread_offsets[thread_id];
  }

  // not used
  std::pair<uint8_t*, size_t> load_file_pinned(const std::string& filename, size_t thread_id) {
    if (shared_buffer == nullptr)
      alloc_future.wait();
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    size_t file_size = static_cast<size_t>(file.tellg());

    if (thread_offsets[thread_id] == 0) {
      size_t offset = global_offset.fetch_add(file_size);
      if (offset + file_size > total_size)
        throw std::runtime_error("Shared buffer is not large enough to accommodate the file.");
      thread_offsets[thread_id] = offset;
    }

    uint8_t* buffer = shared_buffer + thread_offsets[thread_id];

    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(buffer), file_size);

    return std::make_pair(buffer, file_size);
  }

private:
  uint8_t* shared_buffer;
  std::vector<size_t> thread_offsets;
  std::vector<size_t> thread_sizes;
  std::atomic<size_t> global_offset;
  size_t total_size;
  std::future<void> alloc_future;
  std::atomic<bool> is_ready;
};


struct CompressedFile {
  std::string filename;
  std::vector<int64_t> tensor_shape;
  std::string dtype;
  size_t decompressed_size;
  size_t compressed_size;
  bool is_compressed = true;


  // if filename ends with raw, set not compressed otherwise true
  CompressedFile(
      const std::string& filename, 
      const std::vector<int64_t>& tensor_shape, 
      const std::string& dtype, 
      const size_t decompressed_size, 
      const size_t compressed_size
    ): filename(filename), tensor_shape(tensor_shape), dtype(dtype), decompressed_size(decompressed_size), compressed_size(compressed_size){
    // compare returns 0 or the difference between the first non-matching characters
    if (filename.compare(filename.size() - 3, 3, "raw") == 0)
      is_compressed = false;
  }
};


std::vector<int64_t> parse_ints(const std::string& str) {
  // note: tensors can be 0-dimensional, so str can be ""
  std::vector<int64_t> result;
  std::stringstream ss(str);
  std::string dim;
  while (std::getline(ss, dim, ';'))
    if (dim != "")
      result.push_back(std::stoll(dim));
  return result;
}

// change this format for the assignments to be in a separate file for thread partition binning
// nya/thread_to_idx/8.csv, nya/thread_to_idx/32.csv
// just one line forthread indexes separated by spaces


std::pair<std::vector<std::vector<int>>, std::vector<CompressedFile>> load_csv(std::string url) {
  // download file to a buffer and turn it into a stringstream
  auto start = std::chrono::steady_clock::now();
  // download file to /tmp/nya.csv with curl
  // posix_spawn_file_actions_t actions;
  // posix_spawn_file_actions_init(&actions);
  // posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/tmp/nya.csv", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  // posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDERR_FILENO);
  // std::vector<char*> curl_args = { (char*) "-s", (char*) url.c_str() };
  // pid_t pid;
  // if (posix_spawn(&pid, DOWNLOADER_PATH, &actions, NULL, curl_args.data(), NULL) != 0)
  //   throw std::runtime_error("Failed to spawn curl subprocess.");
  // int status;
  // waitpid(pid, &status, 0);
  // if (status != 0)
  //   throw std::runtime_error("Failed to download file: " + url);

  int fd = get_output_fd({ (char*) "-s", (char*) url.c_str() });
  // create stringstream from fd
  std::stringstream file;
  file << fd;
  // parse the file
  return parse_csv(file);  
}


std::pair<std::vector<std::vector<int>>, std::vector<CompressedFile>> load_csv(std::string fname) {
    auto file = std::ifstream(fname);
    return parse_csv(file)
}

std::pair<std::vector<std::vector<int>>, std::vector<CompressedFile>> parse_csv(std::istream file) {
  // filename,tensor_shape,dtype,decompressed_size; for example,
  // 1.gz,224;224,float32,50176
  std::string line;
  std::vector<std::vector<int>> thread_to_idx;
  {
    // the first line is the thread assignments
    std::getline(file, line);
    std::stringstream ss(line);
    std::string thread_str;
    while (std::getline(ss, thread_str, ',')) {
      auto idx = parse_ints(thread_str);
      thread_to_idx.push_back(std::vector<int>(idx.begin(), idx.end()));
    }
  }
  std::vector<CompressedFile> files;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string filename, tensor_shape_str, dtype, size, compressed_size_str;
    ss >> filename >> tensor_shape_str >> dtype >> size >> compressed_size_str;
    std::vector<int64_t> tensor_shape = parse_ints(tensor_shape_str);
    size_t decompressed_size = std::stoull(size);
    size_t compressed_size = std::stoull(compressed_size_str);
    files.emplace_back(filename, tensor_shape, dtype, decompressed_size, compressed_size);
  }
  return std::make_pair(thread_to_idx, files);
}



class SyncedGdeflateManager: public GdeflateManager {
public:
  SyncedGdeflateManager(int chunk_size, nvcompBatchedGdeflateOpts_t compression_level, cudaStream_t stream, std::string name)
    : GdeflateManager(chunk_size, compression_level, stream, NoComputeNoVerify), stream_(stream), name(name) {}

  // Override the destructor
  ~SyncedGdeflateManager() {
    // Synchronize the stream before the GdeflateManager's destructor is called
    log("SyncedGdeflateManager " + name + " called");
    auto start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    log("~SyncedGdeflateManager " + name + " took " + pprint(std::chrono::steady_clock::now() - start));
  }

private:
  cudaStream_t stream_;
  std::string name;
};

const char* DOWNLOADER_PATH = getenv_str("DOWNLOADER_PATH", "/usr/bin/curl");

int get_output_fd(std::vector<char*> curl_args) {
  /* runs curl command as a subprocess with a large pipe buffer size, returning the pipe fd */
  // use posix_spawn to avoid forking a new process
  int pipefd[2];
  if (pipe(pipefd) != 0)
    throw std::runtime_error("Failed to create pipe for curl subprocess.");
  // check proc/sys/fs/pipe-max-size, if we can't read it default to 1MB
  // cf https://github.com/coreweave/tensorizer/blob/main/tensorizer/_wide_pipes.py#L75
  int pipe_max_size;
  {
    std::ifstream file("/proc/sys/fs/pipe-max-size");
    if (file.is_open())
      file >> pipe_max_size;
    else
      pipe_max_size = 1048576;
  }
  // set pipe buffer size
  if (fcntl(pipefd[0], F_SETPIPE_SZ, pipe_max_size) == -1)
    throw std::runtime_error("Failed to set pipe buffer size.");
  // set up posix_spawn
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
  posix_spawn_file_actions_addclose(&actions, pipefd[0]);
  posix_spawn_file_actions_addclose(&actions, pipefd[1]);
  // make sure curl stderr goes to our stderr
  posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDERR_FILENO);
  // spawn curl
  pid_t pid;
  // custom curl that doesn't network backpressure on full pipe
  if (posix_spawn(&pid, DOWNLOADER_PATH, &actions, NULL, curl_args.data(), NULL) != 0)
    throw std::runtime_error("Failed to spawn curl subprocess.");
  // close the write end of the pipe
  close(pipefd[1]);
  // return the read end of the pipe
  return pipefd[0];
}


using ms_t = std::chrono::milliseconds;

int NUM_THREADS = getenv("NUM_THREADS", std::thread::hardware_concurrency());

size_t CHUNK_SIZE = 1 << getenv("CHUNK_SIZE", 20);
int TENSOR_EARLY = getenv("TENSOR_EARLY", 0);
int SYNC_DECOMPRESS = getenv("SYNC_DECOMPRESS", 0);
int SYNC_FREE = getenv("SYNC_FREE", 0);
int NUM_CIRCLE_BUFFERS = getenv("NUM_CIRCLE_BUFFERS", 4);

int WILLNEED = getenv("WILLNEED", 0);
int SEQUENTIAL = getenv("SEQUENTIAL", 0);

size_t UNPINNED_JOBS = getenv("UNPINNED_JOBS", 2);
int UNPINNED_THREADS = getenv("UNPINNED_THREADS", 8);
int DOWNLOAD = getenv("DOWNLOAD", 0);


std::vector<torch::Tensor> batch_decompress(
  const std::vector<CompressedFile>& files,
  const std::vector<std::vector<int>>& thread_to_idx
) {
  auto start_time = std::chrono::steady_clock::now();

  if (files.size() == 0)
    throw std::invalid_argument("Input vector should be non-empty.");

  int num_files = static_cast<int>(files.size());
  int num_threads = std::min(num_files, NUM_THREADS);

  if (thread_to_idx.size() != (size_t)num_threads)
    throw std::invalid_argument("thread_to_idx.size() must be equal to NUM_THREADS, got " + std::to_string(thread_to_idx.size()) + " and " + std::to_string(num_threads));

  ms_t total_copy_time = std::chrono::milliseconds::zero();
  ms_t total_decomp_time = std::chrono::milliseconds::zero();
  ms_t total_read_time = std::chrono::milliseconds::zero();

  std::vector<std::future<std::tuple<ms_t, ms_t, ms_t>>> futures;
  std::vector<torch::Tensor> tensors(num_files);

  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
  // ampere: 128 concurrent kernels ... (3090, A30, etc are 8.6). v100 is 7.0, but still 128

  int num_streams = getenv("NUM_STREAMS", 128);
  if (num_streams < num_threads)
    throw std::invalid_argument("NUM_STREAMS must be >= NUM_THREADS, got " + std::to_string(num_streams) + " and " + std::to_string(num_threads));
  int streams_per_thread = num_streams / num_threads;
  std::vector<std::vector<cudaStream_t>> streams = std::vector<std::vector<cudaStream_t>>(num_threads, std::vector<cudaStream_t>(streams_per_thread));

  log("Using " + std::to_string(num_threads) + " threads and " + std::to_string(streams_per_thread) + " streams per thread for " + std::to_string(num_files) + " files");

  // initialize the primary context 
  {auto start = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaSetDevice(0));
  debug("cudaSetDevice to initialize primary context took " + pprint(std::chrono::steady_clock::now() - start));}

  // size_t total_file_size = getenv("TOTAL_FILE_SIZE", 0);
  // assert(total_file_size > 0);

  size_t total_buffer_size = num_threads * CHUNK_SIZE * NUM_CIRCLE_BUFFERS;

  // this is a significant bottleneck
  // initialize as a pointer to avoid later implicit destruction
  FileLoader* file_loader = new FileLoader(total_buffer_size, num_threads);
  std::vector<cudaEvent_t> thread_copy_done(num_threads, nullptr);
  std::vector<std::vector<std::shared_ptr<nvcomp::nvcompManagerBase>>> thread_managers(num_threads);


  for (int thread_id = 0; thread_id < num_threads; thread_id++) {
    auto indexes = thread_to_idx[thread_id];

    futures.emplace_back(std::async(std::launch::async, [indexes, thread_id, &streams, &tensors, &files, &streams_per_thread, &thread_copy_done, &thread_managers, file_loader]() {
      log("started thread " + std::to_string(thread_id));

      // just do big pipe curl subproc and eventually replace it with our own client with asio


      FILE* curl_file = nullptr;
      if (DOWNLOAD) {
        std::vector<char*> curl_args = { (char*) "-s"};
        if (DEBUG)
          curl_args.push_back((char*) "-v");
        for (auto idx : indexes)
          curl_args.push_back(const_cast<char*>(files[idx].filename.c_str()));
        curl_args.push_back(NULL);  // NULL terminate the arguments list
        int curl_fd = get_output_fd(curl_args);
        // we really need to clean up here somehow
        curl_file = fdopen(curl_fd, "r");
      }

      auto thread_start = std::chrono::steady_clock::now();
      CUDA_CHECK(cudaSetDevice(0)); // this sets the cuda context

      std::chrono::microseconds thread_copy_time, thread_decomp_time = std::chrono::milliseconds::zero();
      std::chrono::nanoseconds thread_read_time = std::chrono::nanoseconds::zero();
      std::vector<std::shared_ptr<nvcomp::nvcompManagerBase>> managers(streams_per_thread, nullptr);
      int64_t total_decompressed_size = 0;

      // uint8_t* scratch_buffer = nullptr;
      // size_t scratch_buffer_size = 0;

      auto create_stream_begin = std::chrono::steady_clock::now();

      for (int stream_id = 0; stream_id < streams_per_thread; stream_id++)
        CUDA_CHECK(cudaStreamCreate(&streams[thread_id][stream_id]));

      log(std::to_string(thread_id) + ": creating streams took " + pprint(std::chrono::steady_clock::now() - create_stream_begin));
      int unpinned_threads = streams.size() - UNPINNED_THREADS;

      std::vector<cudaEvent_t> circle_done_events(NUM_CIRCLE_BUFFERS, nullptr);

      for (size_t job_number = 0; job_number < indexes.size(); job_number++) {
        int i = indexes[job_number];
        auto prefix = "thread " + std::to_string(thread_id) + ", tensor " + std::to_string(i) + " (job " + std::to_string(job_number) + "): ";
        auto file_start_time = std::chrono::steady_clock::now();

        cudaStream_t stream = streams[thread_id][job_number % streams_per_thread];

        // convert stream to hex string for logging
        std::string stream_name = pprint(stream);

        total_decompressed_size += files[i].decompressed_size;

        uint8_t* host_compressed_data;
        size_t input_buffer_len;
        FILE* file = nullptr;
        int fd = -1;
        // if curl_file is not nullptr, we are using curl to download the file and not locally
        if (curl_file != nullptr) {
          input_buffer_len = files[i].compressed_size;
          file = curl_file;
        } else {
          // std::ifstream file = std::ifstream(files[i].filename, std::ios::binary | std::ios::ate);
          // input_buffer_len = static_cast<size_t>(file.tellg());
          // file.seekg(0, std::ios::beg);
          fd = open(files[i].filename.c_str(), O_RDONLY);
          if (fd == -1)
            throw std::invalid_argument("Could not open file " + files[i].filename);
          file = fdopen(fd, "r");
          if (file == nullptr)
            throw std::invalid_argument("Could not open file " + files[i].filename);

          input_buffer_len = lseek(fd, 0, SEEK_END);
          if (input_buffer_len == (size_t)-1)
            throw std::invalid_argument("Could not seek to end of file " + files[i].filename);
          if (input_buffer_len == 0)
            throw std::invalid_argument("File " + files[i].filename + " is empty");
          if (input_buffer_len > files[i].decompressed_size)
            throw std::invalid_argument("File " + files[i].filename + " is bigger than decompressed size" + pprint(input_buffer_len) + " vs " + pprint(files[i].decompressed_size));
          lseek(fd, 0, SEEK_SET);
        }


        debug(prefix + "allocating device memory with stream " + stream_name);
        uint8_t* comp_buffer;
        // if this is an uncompressed tensor, we want comp_buffer to already be the tensor pointer
        if (!files[i].is_compressed) {
          tensors[i] = make_tensor(files[i].tensor_shape, files[i].dtype);
          comp_buffer = static_cast<uint8_t*>(tensors[i].data_ptr());
        } else {
          debug_malloc(prefix + "allocating " + pprint(input_buffer_len) + "compressed device memory");
          CUDA_CHECK(cudaMallocAsync(&comp_buffer, input_buffer_len, stream));
          if (!comp_buffer)
            throw std::runtime_error("Could not allocate device memory for compressed data");
        }
        auto copy_done = thread_copy_done[thread_id];
        if (copy_done != nullptr) {
          auto start = std::chrono::steady_clock::now();
          CUDA_CHECK(cudaEventSynchronize(copy_done)); // wait for previous copy to finish before changing host_compressed_data
          log(prefix + "before using host_compressed_data, waiting for previous copy took " + pprint(std::chrono::steady_clock::now() - start));
        }
        else {
          debug(prefix + "created event");
          CUDA_CHECK(cudaEventCreateWithFlags(&thread_copy_done[thread_id], cudaEventDisableTiming));
          copy_done = thread_copy_done[thread_id];
        }

        auto copy_begin = std::chrono::steady_clock::now();
        debug(prefix + "copying to device with stream " + stream_name);
        size_t already_read = 0;
        int chunks = 0;
        std::string copy_message = "";


        if (i != 0 && job_number < UNPINNED_JOBS && thread_id > unpinned_threads && !DOWNLOAD) {
          auto start = std::chrono::steady_clock::now();
          std::tie(host_compressed_data, input_buffer_len) = load_file_wrapper(files[i].filename);
          thread_read_time += (std::chrono::steady_clock::now() - start);
          CUDA_CHECK(cudaMemcpyAsync(comp_buffer, host_compressed_data, input_buffer_len, cudaMemcpyDefault, stream));
          copy_message = "unpinned ";
          // log(prefix + "NOT PINNED copied " + pprint(input_buffer_len) + " bytes to device in " + pprint(copy_time) + " (total " + pprint_throughput(input_buffer_len, copy_time) + ")");
        } else {
          if (SEQUENTIAL && !DOWNLOAD)
            posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
          int num_buffers = std::min(NUM_CIRCLE_BUFFERS, (int)((input_buffer_len + CHUNK_SIZE - 1) / CHUNK_SIZE));
          host_compressed_data = file_loader->get_buffer(CHUNK_SIZE * NUM_CIRCLE_BUFFERS, thread_id);
          if (host_compressed_data == nullptr)
            throw std::runtime_error("Could not allocate host memory for compressed data");

          // vector of 4 buffers, each pointing to 1/4 offsets into our buffer
          std::vector<uint8_t*> host_buffers(num_buffers);
          for (int i = 0; i < num_buffers; i++)
            host_buffers[i] = host_compressed_data + i * CHUNK_SIZE;

          std::chrono::microseconds sync_time(0);
          int buffer_id = 0;
          while (already_read < input_buffer_len) {
            size_t to_read = std::min(CHUNK_SIZE, input_buffer_len - already_read);
            if (WILLNEED && !DOWNLOAD) 
              posix_fadvise64(fd, already_read, to_read, POSIX_FADV_WILLNEED);
            buffer_id = chunks % num_buffers;
            if (circle_done_events[buffer_id] == nullptr) {
              debug(prefix + "created event");
              CUDA_CHECK(cudaEventCreateWithFlags(&circle_done_events[buffer_id], cudaEventDisableTiming));
            }
            else {
              auto start = std::chrono::steady_clock::now();
              CUDA_CHECK(cudaEventSynchronize(circle_done_events[buffer_id])); // wait for previous copy to finish before changing host_compressed_data
              sync_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start);
            }
            // replace this part with reading from a connection, or an existing fully downloaded buffer
            auto start = std::chrono::steady_clock::now();
            auto fread_result = fread(reinterpret_cast<char*>(host_buffers[buffer_id]), to_read, 1, file);
            thread_read_time += (std::chrono::steady_clock::now() - start);
            if (fread_result != 1) {
              perror("freading file");
              throw std::runtime_error("Could not read file " + files[i].filename + " (size " + pprint(input_buffer_len) + "): " + std::to_string(fread_result) + ", eof: " + std::to_string(feof(file)));
            }
            CUDA_CHECK(cudaMemcpyAsync(comp_buffer + already_read, host_buffers[buffer_id], to_read, cudaMemcpyDefault, stream));
            already_read += to_read;
            chunks++;
            CUDA_CHECK(cudaEventRecord(circle_done_events[buffer_id], stream));
          }
          thread_copy_done[thread_id] = circle_done_events[buffer_id];
          log(prefix + "took " + pprint(sync_time) + " to sync " + std::to_string(chunks) + " chunks");
        }
        std::chrono::microseconds copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - copy_begin);
        std::string pretty_chunk_size = pprint(std::min(CHUNK_SIZE, input_buffer_len), 0);
        log(prefix + copy_message + "copied " + std::to_string(chunks) + " " + pretty_chunk_size + " chunks in " + pprint(copy_time) + " (" + pprint_throughput(input_buffer_len, copy_time) + ")");
        if (!DOWNLOAD)
          fclose(file);

        // if (job_number == indexes.size() - 1) {
        //   log(prefix + "scheduling release of file_loader on stream " + std::to_string(stream_int) + " (job " + std::to_string(job_number) + ")");
        //   CUDA_CHECK(cudaLaunchHostFunc(stream, FileLoader::release, file_loader));
        // }
        if (files[i].is_compressed){
          if (TENSOR_EARLY)
            tensors[i] = make_tensor(files[i].tensor_shape, files[i].dtype);
          auto decomp_nvcomp_manager = managers[job_number % streams_per_thread];

          if (!decomp_nvcomp_manager) {
            auto create_manager_begin = std::chrono::steady_clock::now();
            std::string name = "thread-" + std::to_string(thread_id) + "-stream-" + stream_name;
            decomp_nvcomp_manager = std::make_shared<SyncedGdeflateManager>(1 << 16, nvcompBatchedGdeflateDefaultOpts, stream, name); // 1 << 16 is 64KB, 0 is fast compression
            managers[job_number % streams_per_thread] = decomp_nvcomp_manager;
            log("created manager in " + pprint(std::chrono::steady_clock::now() - create_manager_begin) + " for stream " + stream_name + " (job " + std::to_string(job_number) + ")");
          }

          auto config_begin = std::chrono::steady_clock::now();

          // this syncs the stream
          DecompressionConfig decomp_config = decomp_nvcomp_manager->configure_decompression(comp_buffer);
          auto decomp_begin = std::chrono::steady_clock::now();
          debug(prefix + "configuring decomp took " + pprint(decomp_begin - config_begin) + ", decompressing");

          // auto new_size = decomp_nvcomp_manager.get()->get_required_scratch_buffer_size();
          // // cudaMemPoolCreate(), cudaMallocFromPoolAsync()
          // // auto scratch_buffer = std::make_unique<uint8_t[]>(size);        
          // if (scratch_buffer == nullptr || scratch_buffer_size < new_size) {
          //   if (scratch_buffer != nullptr) {
          //     log(prefix + "freeing smaller scratch buffer of size " + std::to_string(scratch_buffer_size));
          //     CUDA_CHECK(cudaFreeAsync(scratch_buffer, stream));
          //   }
          //   scratch_buffer_size = new_size;
          //   log(prefix + "allocating new scratch buffer of size " + std::to_string(scratch_buffer_size) + " for decompressed size " + std::to_string(files[i].decompressed_size));
          //   CUDA_CHECK(cudaMallocAsync(&scratch_buffer, scratch_buffer_size, stream));
          // }
          // decomp_nvcomp_manager->set_scratch_buffer(scratch_buffer);
          if (!TENSOR_EARLY)
            tensors[i] = make_tensor(files[i].tensor_shape, files[i].dtype);

          bool safe = tensors[i].is_contiguous();
          if (!safe) {
            std::cerr << "Tensor " << i << " is not contiguous!!! something bad has happened" << std::endl;
            throw std::runtime_error("Tensor " + std::to_string(i) + " is not contiguous");
          }
          auto tens = tensors[i];
          auto ptr = tens.data_ptr();
          auto dest = static_cast<uint8_t*>(ptr);

          // if this tensor was uncompressed, don't decompress, but also don't free comp_buffer

          try {
            decomp_nvcomp_manager->decompress(dest, comp_buffer, decomp_config);
          }
          catch (const std::exception& e) {
            log(prefix + "exception: " + std::string(e.what()));
            log(prefix + "cuda error: " + std::string(cudaGetErrorString(cudaGetLastError())));
            log(prefix + "cuda stream error: " + std::string(cudaGetErrorString(cudaStreamQuery(stream))));
            throw e;
          }

          if (SYNC_DECOMPRESS)
            CUDA_CHECK(cudaStreamSynchronize(stream));
          // decomp_nvcomp_manager->decompress(static_cast<uint8_t*>(tensors[i].data_ptr()), comp_buffer, decomp_config);
          ms_t decomp_time = std::chrono::duration_cast<ms_t>(std::chrono::steady_clock::now() - decomp_begin);

          log(prefix + "decompressed in " + pprint(decomp_time) + ", freeing with stream " + stream_name + "");
          // need to not free comp_buffer until the decompression is complete, so we should 
          CUDA_CHECK(cudaFreeAsync(comp_buffer, stream));
          if (SYNC_FREE)
            CUDA_CHECK(cudaStreamSynchronize(stream));
          thread_decomp_time += decomp_time; // shrug
        }
        auto file_elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - file_start_time).count();

        log(prefix + "processed in " + std::to_string(file_elapsed_time) + " ms");
        thread_copy_time += copy_time;
        
      }
      auto thread_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - thread_start);
      auto thread_elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(thread_elapsed);

      int throughput = std::round((float)total_decompressed_size / (float)thread_elapsed.count() * 1000 / 1024.0f / 1024.0f);
      log("thread " + std::to_string(thread_id) + " processed " + std::to_string(total_decompressed_size / 1024) + "kb in " + std::to_string(thread_elapsed.count()) + " ms (" + std::to_string(throughput) + " MB/s) - " + std::to_string(indexes.size()) + " files");
      // std::this_thread::sleep_for(std::chrono::milliseconds(getenv("SLEEP", 3)*1000));

      
      // find the file that's all the small tensors, decompress, then create tensors from that with from_blob
      // read the offsets from a file

      // thread_managers[thread_id] = managers;
      return std::make_tuple(std::chrono::duration_cast<ms_t>(thread_copy_time), std::chrono::duration_cast<ms_t>(thread_decomp_time), std::chrono::duration_cast<ms_t>(thread_read_time));
      }));
    // sleep to give the first threads thread a chance to start 
    std::this_thread::sleep_for(std::chrono::milliseconds(getenv("SLEEP", 3)));
  }

  // for (auto &f : futures) f.wait();
  for (int i = 0; i < (int)futures.size(); i++) {
    ms_t thread_copy_time, thread_decomp_time, thread_read_time;
    log("waiting for future " + std::to_string(i));
    std::tie(thread_copy_time, thread_decomp_time, thread_read_time) = futures[i].get();
    debug("got future " + std::to_string(i));
    total_copy_time += thread_copy_time;
    total_decomp_time += thread_decomp_time;
    total_read_time += thread_read_time;
  }

  // for (auto managers : thread_managers)
  //   for (auto manager : managers)
  //     manager.reset();

  for (auto& event : thread_copy_done)
    if (event != nullptr) {
      CUDA_CHECK(cudaEventSynchronize(event));
      CUDA_CHECK(cudaEventDestroy(event));
    }
    else
      std::cerr << "a copy_done event is null, a thread must have failed to start or not done any work" << std::endl;

  delete file_loader;

  auto sync_start = std::chrono::steady_clock::now();
  for (auto& thread_streams : streams)
    for (auto& stream : thread_streams)
      CUDA_CHECK(cudaStreamSynchronize(stream));
  for (auto& thread_streams : streams)
    for (auto& stream : thread_streams)
      CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  log("Sync time: " + pprint(std::chrono::steady_clock::now() - sync_start));


  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  log("Total processing time: " + std::to_string(elapsed_time) + " ms for " + std::to_string(num_files) + " tensors on " + std::to_string(num_threads) + " threads, " + std::to_string(num_streams) + " streams");
  log("Total copy time: " + std::to_string(total_copy_time.count()) + "[ms], total decomp time: " + std::to_string(total_decomp_time.count()) + "[ms], total read time: " + std::to_string(total_read_time.count()) + "[ms]");
  log("Average copy time per file: " + std::to_string((total_copy_time / num_files).count()) + "[ms], average decomp time per file: " + std::to_string((total_decomp_time / num_files).count()) + "[ms]");
  log("Average copy time per thread: " + std::to_string((total_copy_time / num_threads).count()) + "[ms], average decomp time per thread: " + std::to_string((total_decomp_time / num_threads).count()) + "[ms]");

  std::ofstream file("/tmp/stats.json", std::ios::app);
  file << "{\"elapsed_time\":" << elapsed_time << ",\"num_files\":" << num_files;
  file << ",\"num_threads\":" << num_threads << ",\"num_streams\":" << num_streams;
  file << ",\"total_copy_time\":" << total_copy_time.count() << ",\"total_decomp_time\":" << total_decomp_time.count();
  file << ",\"total_file_size\":" << total_buffer_size << ",\"chunk_size\":" << CHUNK_SIZE;
  file << ",\"sleep\":" << getenv("SLEEP", 3) << ",\"total_read_time\":" << total_read_time.count();
  file << ", \"name\":\"" << (std::getenv("NAME") ? std::getenv("NAME") : "unknown") << "\"";
  file << "}" << std::endl;
  file.close();

  return tensors;
}

class AsyncDecompressor {
private:
  std::future<std::vector<torch::Tensor>> future;

public:
  bool done = false;
  bool failed = false;
  std::string error;

  AsyncDecompressor(std::string fname) {
    std::vector<CompressedFile> files;
    std::vector<std::vector<int>> thread_to_idx;
    std::tie(thread_to_idx, files) = load_csv(fname);
    // ideally do some more validation

    size_t num_threads = std::min(static_cast<int>(files.size()), NUM_THREADS);
    if (thread_to_idx.size() != num_threads) {
      error = "thread_to_idx.size() must be equal to NUM_THREADS, got " + std::to_string(thread_to_idx.size()) + " and " + std::to_string(num_threads);
      std::cerr << error << std::endl;
      failed = true;
    }
    else {
      log("starting batch_decompress future");
      future = std::async(std::launch::async, batch_decompress, files, thread_to_idx);
    }
  }

  std::vector<torch::Tensor> get() {
    log("called AsyncDecompressor::get()");
    if (failed) throw std::runtime_error(error);
    if (done) throw std::runtime_error("get() called on a decompressor that is already done");
    try {
      auto tensors = future.get();
      done = true;
      return tensors;
    }
    catch (const std::exception& e) {
      failed = true;
      error = e.what();
      std::cerr << error << std::endl;
      throw e;
    }
  }

  ~AsyncDecompressor() {
    log("called AsyncDecompressor destructor");
  }
};


PYBIND11_MODULE(_nyacomp, m) {
  m.doc() = R"pbdoc(python bindings for nvcomp with torch)pbdoc";

  m.def("compress", &compress, R"pbdoc(compress bytes to a file)pbdoc", py::arg("data"), py::arg("filename"));

  m.def("decompress", &decompress, "decompress to a new tensor", py::arg("filename"), py::arg("shape"), py::arg("dtype"));

  m.def("batch_decompress", &batch_decompress, "decompress tensors with a threadpool", py::arg("files"), py::arg("assignments"), py::call_guard<py::gil_scoped_release>());

  py::class_<CompressedFile>(m, "CompressedFile")
    // filename, shape, dtype, buffer_size, compressed_size
    .def(py::init<const std::string&, const std::vector<int64_t>&, const std::string&, const size_t, const size_t>());

  py::class_<AsyncDecompressor>(m, "AsyncDecompressor")
    .def(py::init<const std::string&>(), py::call_guard<py::gil_scoped_release>())
    .def("get", &AsyncDecompressor::get, py::call_guard<py::gil_scoped_release>());

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
