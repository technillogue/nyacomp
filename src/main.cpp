#include <assert.h>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <iostream>
#include <signal.h>
#include <spawn.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
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
}

int getenv(const char* name, int default_value) {
  auto value = std::getenv(name);
  return (value == nullptr || value[0] == '\0') ? default_value : std::stoi(value);
}

const bool DEBUG = getenv("DEBUG", 0);
const bool SILENT = getenv("SILENT", 0);
const bool DEBUG_MALLOC = getenv("DEBUG_MALLOC", 0);
const bool DEBUG_EVENT = getenv("DEBUG_SYNC", 0);

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

void debug_event(const std::string& msg) {
  if (DEBUG_EVENT)
    std::cout << msg << std::endl;
}


// size_t round_up_kb(size_t size) { return (size + 1023) & -1024; }

template<typename T>
std::string generic_pprint_hex(T t) {
  std::stringstream ss;
  ss << std::hex << t;
  return ss.str();
}

std::string pprint(cudaStream_t stream) {
  return generic_pprint_hex(stream);
}

std::string pprint(cudaEvent_t event) {
  return generic_pprint_hex(event);
}

std::string pprint(uint8_t* ptr) {
  return generic_pprint_hex(ptr);
}


std::string pprint(std::vector<char*> vec) {
  std::stringstream ss;
  ss << "[";
  for (auto& s : vec)
    ss << s << ", ";
  ss << "]";
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
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
  if (ns == 0)
    return pprint(bytes) + "/0s";
  return pprint(bytes * 1e9 / ns) + "/s";
}

const bool NOTIME = getenv("NOTIME", 0);

std::chrono::steady_clock::time_point maybe_now() {
  if (NOTIME)
    return std::chrono::steady_clock::time_point();
  return std::chrono::steady_clock::now();
}

std::vector<uint8_t> load_file(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error("Failed to open file: " + filename);

  size_t file_size = static_cast<size_t>(file.tellg());
  std::vector<uint8_t> buffer(file_size);

  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size))
    throw std::runtime_error("Failed to read file: " + filename);
  debug("read " + pprint(file_size) + " bytes from " + filename);

  return buffer;
}

std::pair<uint8_t*, size_t> load_file_wrapper(const std::string& filename) {
  auto buffer = load_file(filename);
  return std::make_pair(buffer.data(), buffer.size());
}


void widen_pipe(int fd) {
  // check proc/sys/fs/pipe-max-size, if we can't read it default to 1MB
  // cf https://github.com/coreweave/tensorizer/blob/main/tensorizer/_wide_pipes.py#L75
  int max_pipe_size;
  std::ifstream file("/proc/sys/fs/pipe-max-size");
  std::string is_default = "";
  if (file.is_open()) {
    file >> max_pipe_size;
  } else {
    max_pipe_size = 1048576;
    is_default = " (default, couldn't read /proc/sys/fs/pipe-max-size)";
  }
  int capacity = fcntl(fd, F_SETPIPE_SZ, max_pipe_size);
  if (capacity == -1) {
    if (errno == EINVAL)
      throw std::runtime_error("EINVAL: Failed to set pipe size, invalid argument (are we on linux?).");
    else if (errno == EPERM) 
      throw std::runtime_error("EPERM: Permission denied to set pipe size to " + std::to_string(max_pipe_size) + is_default + ". Is /proc/sys/fs/pipe-max-size set to a larger value? ");
    else if (errno == EBUSY)
      throw std::runtime_error("EBUSY: Failed to set pipe size, pipe is busy.");
    else
      throw std::runtime_error("Failed to set pipe size, unknown error: " + std::string(strerror(errno)));
  }
}

const char* DOWNLOADER_PATH = getenv_str("DOWNLOADER_PATH", "/usr/bin/curl");
bool DOWNLOAD_LOG = getenv("DOWNLOAD_LOG", 0);
bool SKIP_SETPIPE_SZ = getenv("SKIP_SETPIPE_SZ", 0);

class DownloadProc {
 public:
  /* 
    download urls in a subprocess, return a file pointer to the pipe
    posix_spawn is used to avoid forking
    pipe buffer is widened
  */
  FILE* download(std::vector<std::string> urls) {
    std::vector<char*> curl_args = { (char*) "-s"};
    curl_args.reserve(urls.size() + 2);
    if (DEBUG)
      curl_args.push_back((char*) "-v");
    for (auto& url : urls)
      curl_args.push_back(const_cast<char*>(url.c_str()));
    curl_args.push_back(NULL);  // NULL terminate the arguments list

    int pipefd[2];
    if (pipe(pipefd) != 0)
      throw std::runtime_error("Failed to create pipe for curl subprocess.");
    if (!SKIP_SETPIPE_SZ)
      widen_pipe(pipefd[0]);
    debug("using downloader path: " + std::string(DOWNLOADER_PATH));
    // set up posix_spawn
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    posix_spawn_file_actions_addclose(&actions, pipefd[1]);
    if (DOWNLOAD_LOG) {
      int log_fd = open("/tmp/downloader.log", O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (log_fd == -1)
        throw std::runtime_error("Failed to open /tmp/downloader.log");
      posix_spawn_file_actions_adddup2(&actions, log_fd, STDERR_FILENO);
      posix_spawn_file_actions_addclose(&actions, log_fd);
    } else
      posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDERR_FILENO);
    // spawn curl or custom curl that doesn't network backpressure on full pipe
    int retcode = posix_spawn(&pid, DOWNLOADER_PATH, &actions, NULL, curl_args.data(), environ);
    if (retcode != 0) {
      debug("downloader args: " + pprint(curl_args));
      throw std::runtime_error("Failed to spawn curl subprocess. Exit code: " + std::to_string(retcode));
    }
    // close write end of the pipe
    close(pipefd[1]);
    // return read end of the pipe
    return fdopen(pipefd[0], "r");
  }

  ~DownloadProc() {
    if (pid == -1)
      return;
    log("killing downloader process");
    kill(pid, SIGTERM);
    waitpid(pid, NULL, 0);
  }

 private:
  pid_t pid = -1;
};


int compress(py::bytes pybytes, const std::string filename, const int algo = 0) {
  std::string bytes_str = pybytes;
  size_t input_buffer_len = bytes_str.size();
  std::vector<uint8_t> uncompressed_data(bytes_str.data(), bytes_str.data() + input_buffer_len);
  debug("working with " + pprint(input_buffer_len) + " uncompressed bytes");

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  uint8_t* device_input_ptrs;
  CUDA_CHECK(cudaMallocAsync(&device_input_ptrs, input_buffer_len, stream));
  CUDA_CHECK(cudaMemcpyAsync(device_input_ptrs, uncompressed_data.data(), input_buffer_len, cudaMemcpyDefault, stream));


  const size_t chunk_size = 1 << 16;
  // DEFLATE is LZ77 dictionary + Huffman entropy coding

  // 0 : high-throughput, low compression ratio (default) // only supported, lolsob
  // 1 : low-throughput, high compression ratio
  // 2 : highest-throughput, entropy-only compression (use for symmetric compression/decompression performance)

  // maybe it's time to rethink this? old testing suggested skipping dictionary could actually give better throughput
  // but 1 also changes some huffman settings that would have been good
  nvcompBatchedGdeflateOpts_t opts = { algo };
  GdeflateManager nvcomp_manager{ chunk_size, opts, stream, ComputeAndVerifyIfPresent };

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
  else if (type_name == "bfloat16")
    return torch::kBFloat16;
  else
    throw std::runtime_error("Unknown type name: " + type_name);
}

torch::Tensor make_tensor(const std::vector<int64_t>& shape, const std::string& dtype) {
  auto options = torch::TensorOptions().dtype(type_for_name(dtype)).device(torch::kCUDA).memory_format(torch::MemoryFormat::Contiguous);
  return torch::empty(shape, options);
}


torch::Tensor decompress(const std::string filename, std::vector<int64_t> shape, std::string dtype) {
  std::vector<uint8_t> compressed_data = load_file(filename);
  size_t input_buffer_len = compressed_data.size();
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

  // std::pair<uint8_t*, size_t> load_file_pinned(const std::string& filename, size_t thread_id) {
  //   if (shared_buffer == nullptr)
  //     alloc_future.wait();
  //   std::ifstream file(filename, std::ios::binary | std::ios::ate);
  //   size_t file_size = static_cast<size_t>(file.tellg());
  //
  //   if (thread_offsets[thread_id] == 0) {
  //     size_t offset = global_offset.fetch_add(file_size);
  //     if (offset + file_size > total_size)
  //       throw std::runtime_error("Shared buffer is not large enough to accommodate the file.");
  //     thread_offsets[thread_id] = offset;
  //   }
  //
  //   uint8_t* buffer = shared_buffer + thread_offsets[thread_id];
  //   file.seekg(0, std::ios::beg);
  //   file.read(reinterpret_cast<char*>(buffer), file_size);
  //   return std::make_pair(buffer, file_size);
  // }

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
    ): filename(filename), tensor_shape(tensor_shape), dtype(dtype), decompressed_size(decompressed_size), compressed_size(compressed_size) {
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

std::pair<std::vector<CompressedFile>, std::vector<std::vector<int>>> parse_csv(std::istream& file) {
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
    try {
      std::stringstream ss(line);
      std::string filename, tensor_shape_str, dtype, size, compressed_size_str;
      ss >> filename >> tensor_shape_str >> dtype >> size >> compressed_size_str;
      std::vector<int64_t> tensor_shape = parse_ints(tensor_shape_str);
      size_t decompressed_size = std::stoull(size);
      size_t compressed_size = std::stoull(compressed_size_str);
      files.emplace_back(filename, tensor_shape, dtype, decompressed_size, compressed_size);
      // print line being parsed in case of error
    } catch (std::exception& e) {
      std::cout << "Error parsing line: " << line << std::endl;
      throw e;
    }

  }
  return std::make_pair(files, thread_to_idx);
}


std::pair<std::vector<CompressedFile>, std::vector<std::vector<int>>> load_csv(std::string fname) {
    auto file = std::ifstream(fname);
    return parse_csv(file);
}

std::pair<std::vector<CompressedFile>, std::vector<std::vector<int>>> load_remote_csv(std::string url) {
  // download file to a buffer and turn it into a stringstream
  auto start = std::chrono::steady_clock::now();
  DownloadProc downloader;
  FILE* output = downloader.download({url});
  debug("launched csv download, proceeding with parsing");
  // create stringstream from fd
  std::stringstream file;
  // read the entire file
  while (true) {
    char buf[1024];
    size_t bytes_read = fread(buf, 1, sizeof(buf), output);
    if (bytes_read == 0)
      break;
    file.write(buf, bytes_read);
  }

  file.seekg(0, std::ios::end);
  auto size = file.tellg();
  log("read " + pprint(size) + " bytes from remote csv " + url + " in " + pprint(std::chrono::steady_clock::now() - start));
  file.seekg(0, std::ios::beg);


  if (DEBUG) {
    std::cout << file.str(); 
    log("dumped csv file to stdout, resetting file pointer");
    file.seekg(0, std::ios::beg);
  }
  // parse the file
  return parse_csv(file); 
}

int NUM_THREADS = getenv("NUM_THREADS", std::thread::hardware_concurrency());
int DOWNLOAD = getenv("DOWNLOAD", 0);
int REMOTE_CSV = getenv("REMOTE_CSV", DOWNLOAD);


std::pair<std::vector<CompressedFile>, std::vector<std::vector<int>>> load_any_csv(std::string fname) {
    std::vector<CompressedFile> files;
    std::vector<std::vector<int>> thread_to_idx;
    if (DOWNLOAD && REMOTE_CSV)
        std::tie(files, thread_to_idx) = load_remote_csv(fname);
    else
        std::tie(files, thread_to_idx) = load_csv(fname);
    // ideally do some more validation
    size_t num_threads = std::min(static_cast<int>(files.size()), NUM_THREADS);
    if (thread_to_idx.size() != num_threads) {
      auto error = "thread_to_idx.size() must be equal to NUM_THREADS, got " + std::to_string(thread_to_idx.size()) + " and " + std::to_string(num_threads);
      std::cerr << error << std::endl;
      throw std::invalid_argument(error);
    }
    return std::make_pair(files, thread_to_idx);
}

class SyncedGdeflateManager: public GdeflateManager {
public:
  SyncedGdeflateManager(int chunk_size, nvcompBatchedGdeflateOpts_t compression_level, cudaStream_t stream, std::string name)
    : GdeflateManager(chunk_size, compression_level, stream, NoComputeNoVerify), stream_(stream), name(name) {}

  // Override the destructor
  ~SyncedGdeflateManager() {
    // Synchronize the stream before the GdeflateManager's destructor is called
    log("~SyncedGdeflateManager " + name + " called");
    auto start = maybe_now();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto elapsed = maybe_now() - start;
    if (elapsed > std::chrono::milliseconds(0))
      debug("~SyncedGdeflateManager " + name + " took " + pprint(std::chrono::steady_clock::now() - start));
  }

private:
  cudaStream_t stream_;
  std::string name;
};

using ms_t = std::chrono::milliseconds;

struct FileAndFd {
  FILE* file;
  int fd;
  size_t size;
};

FileAndFd open_file(CompressedFile comp_file) {
  auto filename = comp_file.filename;
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1)
    throw std::invalid_argument("Could not open file " + filename);
  FILE* file = fdopen(fd, "r");
  if (file == nullptr)
    throw std::invalid_argument("Could not open file " + filename);
  size_t size = lseek(fd, 0, SEEK_END);
  if (size == (size_t)-1)
    throw std::invalid_argument("Could not seek to end of file " + filename);
  if (size == 0)
    throw std::invalid_argument("File " + filename + " is empty");
  if (size > comp_file.decompressed_size)
    throw std::invalid_argument("File " + filename + " is bigger than decompressed size" + pprint(size) + " vs " + pprint(comp_file.decompressed_size));
  lseek(fd, 0, SEEK_SET);
  return {file, fd, size};
}



size_t CHUNK_SIZE = 1 << getenv("CHUNK_SIZE", 20);
int TENSOR_EARLY = getenv("TENSOR_EARLY", 0);
int SYNC_DECOMPRESS = getenv("SYNC_DECOMPRESS", 0);
int SYNC_FREE = getenv("SYNC_FREE", 0);
int REUSE_COMP_BUFFER = getenv("REUSE_COMP_BUFFER", 0);
int NUM_CIRCLE_BUFFERS = getenv("NUM_CIRCLE_BUFFERS", 4);

int WILLNEED = getenv("WILLNEED", 0);
int SEQUENTIAL = getenv("SEQUENTIAL", 0);

size_t UNPINNED_JOBS = getenv("UNPINNED_JOBS", 2);
int UNPINNED_THREADS = getenv("UNPINNED_THREADS", 8);

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

  log("using " + std::to_string(num_threads) + " threads and " + std::to_string(streams_per_thread) + " streams per thread for " + std::to_string(num_files) + " files");


  std::vector<FILE*> curl_files_for_threads(num_threads, nullptr);
  std::vector<DownloadProc> downloaders(num_threads);
  if (DOWNLOAD) {
    for (size_t thread_id = 0; thread_id < (size_t)num_threads; thread_id++) {
      std::vector<std::string> urls;
      urls.reserve(thread_to_idx[thread_id].size());
      for (auto idx : thread_to_idx[thread_id])
        urls.push_back(files[idx].filename);
      curl_files_for_threads[thread_id] = downloaders[thread_id].download(urls);
    }
  }

  // initialize the primary context 
  // see https://forums.developer.nvidia.com/t/cuda-context-and-threading/26625/6 for horrors untold
  {auto start = std::chrono::steady_clock::now(); 
  CUDA_CHECK(cudaSetDevice(0));
  debug("cudaSetDevice to initialize primary context took " + pprint(std::chrono::steady_clock::now() - start));}

  size_t total_buffer_size = num_threads * CHUNK_SIZE * NUM_CIRCLE_BUFFERS;

  // this is a significant bottleneck
  // initialize as a pointer to avoid later implicit destruction
  FileLoader* file_loader = new FileLoader(total_buffer_size, num_threads);
  std::vector<cudaEvent_t> thread_copy_done(num_threads, nullptr);
  std::vector<std::vector<std::shared_ptr<nvcomp::nvcompManagerBase>>> thread_managers(num_threads);


  for (int thread_id = 0; thread_id < num_threads; thread_id++) {
    auto indexes = thread_to_idx[thread_id];
    FILE* curl_file = curl_files_for_threads[thread_id];

    futures.emplace_back(std::async(std::launch::async, [indexes, thread_id, curl_file, &streams, &tensors, &files, &streams_per_thread, &thread_copy_done, &thread_managers, file_loader]() {
      log("started thread " + std::to_string(thread_id));

      auto thread_start = std::chrono::steady_clock::now();
      CUDA_CHECK(cudaSetDevice(0)); // this sets the cuda context

      std::chrono::microseconds thread_copy_time, thread_decomp_time = std::chrono::milliseconds::zero();
      std::chrono::nanoseconds thread_read_time = std::chrono::nanoseconds::zero();
      std::vector<std::shared_ptr<nvcomp::nvcompManagerBase>> managers(streams_per_thread, nullptr);
      int64_t thread_decompressed_size = 0;

      // uint8_t* scratch_buffer = nullptr;
      // size_t scratch_buffer_size = 0;

      auto create_stream_begin = std::chrono::steady_clock::now();

      for (int stream_id = 0; stream_id < streams_per_thread; stream_id++)
        CUDA_CHECK(cudaStreamCreate(&streams[thread_id][stream_id]));

      log(std::to_string(thread_id) + ": creating streams took " + pprint(std::chrono::steady_clock::now() - create_stream_begin));
      int unpinned_threads = streams.size() - UNPINNED_THREADS;

      std::vector<cudaEvent_t> circle_done_events(NUM_CIRCLE_BUFFERS, nullptr);

      // reuse the same comp_buffer for all jobs 
      uint8_t* saved_comp_buffer = nullptr;
      size_t saved_comp_buffer_size = 0;

      for (size_t job_number = 0; job_number < indexes.size(); job_number++) {
        int i = indexes[job_number];
        auto prefix = "thread " + std::to_string(thread_id) + ", tensor " + std::to_string(i) + " (job " + std::to_string(job_number) + "): ";
        auto file_start_time = std::chrono::steady_clock::now();

        cudaStream_t stream = streams[thread_id][job_number % streams_per_thread];

        // convert stream to hex string for logging
        std::string stream_name = pprint(stream);

        thread_decompressed_size  += files[i].decompressed_size;

        uint8_t* host_compressed_data;

        size_t input_buffer_len;
        FILE* file = nullptr;
        int fd = -1;
        // if curl_file is not nullptr, we are using curl to download the file and not locally
        if (curl_file != nullptr) {
          file = curl_file;
          input_buffer_len = files[i].compressed_size;
        } else {
          FileAndFd file_and_fd = open_file(files[i]);
          file = file_and_fd.file;
          fd = file_and_fd.fd;
          input_buffer_len = file_and_fd.size;
        }


        debug(prefix + "allocating device memory with stream " + stream_name);
        uint8_t* comp_buffer;
        // if this is an uncompressed tensor, we want comp_buffer to already be the tensor pointer
        if (!files[i].is_compressed) {
          tensors[i] = make_tensor(files[i].tensor_shape, files[i].dtype);
          comp_buffer = static_cast<uint8_t*>(tensors[i].data_ptr());
        } else if (REUSE_COMP_BUFFER) {
          if (saved_comp_buffer != nullptr && saved_comp_buffer_size < input_buffer_len) {
            debug_malloc(prefix + "previous comp_buffer size " + pprint(saved_comp_buffer_size) + " is smaller than current " + pprint(input_buffer_len) + ", freeing " + pprint(saved_comp_buffer));
            CUDA_CHECK(cudaFree(saved_comp_buffer));
            saved_comp_buffer_size = 0;
          }          
          if (saved_comp_buffer == nullptr) {
            debug_malloc(prefix + "allocating " + pprint(input_buffer_len) + " reusable compressed device memory");
            CUDA_CHECK(cudaMallocAsync(&saved_comp_buffer, input_buffer_len, stream));
            if (!saved_comp_buffer)
              throw std::runtime_error("Could not allocate device memory for compressed data");
            saved_comp_buffer_size = input_buffer_len;
          }
          comp_buffer = saved_comp_buffer;
        } else {
          debug_malloc(prefix + "allocating " + pprint(files[i].compressed_size) + " compressed device memory");
          CUDA_CHECK(cudaMallocAsync(&comp_buffer, files[i].compressed_size, stream));
        }

        auto copy_done = thread_copy_done[thread_id];
        if (copy_done != nullptr) {
          auto start = std::chrono::steady_clock::now();
          debug_event(prefix + "synchronizing copy_done event " + pprint(copy_done));
          CUDA_CHECK(cudaEventSynchronize(copy_done)); // wait for previous copy to finish before changing host_compressed_data
          debug(prefix + "before using host_compressed_data, waiting for previous copy took " + pprint(std::chrono::steady_clock::now() - start));
        }
        else {
          CUDA_CHECK(cudaEventCreateWithFlags(&thread_copy_done[thread_id], cudaEventDisableTiming));
          copy_done = thread_copy_done[thread_id];
          debug_event(prefix + "created event " + pprint(copy_done));
        }

        auto copy_begin = std::chrono::steady_clock::now();
        debug(prefix + "copying to device with stream " + stream_name);
        size_t already_read = 0;
        int chunks = 0;
        std::string copy_message = "";


        if (i != 0 && job_number < UNPINNED_JOBS && thread_id > unpinned_threads && !DOWNLOAD) {
          auto start = maybe_now();
          std::tie(host_compressed_data, input_buffer_len) = load_file_wrapper(files[i].filename);
          thread_read_time += (maybe_now() - start);
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
              CUDA_CHECK(cudaEventCreateWithFlags(&circle_done_events[buffer_id], cudaEventDisableTiming));
              debug_event(prefix + "created event " + pprint(circle_done_events[buffer_id]));
            }
            else {
              auto start = maybe_now();
              debug_event(prefix + "synchronizing circle_done event " + pprint(circle_done_events[buffer_id]));
              CUDA_CHECK(cudaEventSynchronize(circle_done_events[buffer_id])); // wait for previous copy to finish before changing host_compressed_data
              sync_time += std::chrono::duration_cast<std::chrono::microseconds>(maybe_now() - start);
            }
            // replace this part with reading from a connection, or an existing fully downloaded buffer
            auto start = maybe_now();
            auto fread_result = fread(reinterpret_cast<char*>(host_buffers[buffer_id]), to_read, 1, file);
            thread_read_time += (maybe_now() - start);
            if (fread_result != 1) {
              perror(("freading returned " + std::to_string(fread_result) + " instead of 1, error").c_str());
              throw std::runtime_error("Could not read file " + files[i].filename + " (size " + pprint(input_buffer_len) + "): " + std::to_string(fread_result) + ", eof: " + std::to_string(feof(file)));
            }
            CUDA_CHECK(cudaMemcpyAsync(comp_buffer + already_read, host_buffers[buffer_id], to_read, cudaMemcpyDefault, stream));
            already_read += to_read;
            chunks++;
            CUDA_CHECK(cudaEventRecord(circle_done_events[buffer_id], stream));
          }
          thread_copy_done[thread_id] = circle_done_events[buffer_id];
          debug(prefix + "took " + pprint(sync_time) + " to sync " + std::to_string(chunks) + " chunks");
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
        if (files[i].is_compressed) {
          if (TENSOR_EARLY)
            tensors[i] = make_tensor(files[i].tensor_shape, files[i].dtype);
          auto decomp_nvcomp_manager = managers[job_number % streams_per_thread];

          if (!decomp_nvcomp_manager) {
            // std::string name = "thread-" + std::to_string(thread_id) + "-stream-" + stream_name;
            // decomp_nvcomp_manager = create_manager(stream, manager_name, stream_name);
            // managers[job_number % streams_per_thread] = decomp_nvcomp_manager;
            

            /*
            std::shared_ptr <nvcomp::nvcompBatchedGdeflateDecompressor> create_manager(
              cudaStream_t stream, 
              std::string manager_name, 
              std::string stream_name
            ) {
              auto create_manager_begin = std::chrono::steady_clock::now();
              // 1 << 16 is 64KB, 0 is fast compression
              decomp_nvcomp_manager = std::make_shared<SyncedGdeflateManager>(1 << 16, nvcompBatchedGdeflateDefaultOpts, stream, manager_name);
              log("created manager in " + pprint(std::chrono::steady_clock::now() - create_manager_begin) + " for stream " + stream_name + " (job " + std::to_string(job_number) + ")");
              return decomp_nvcomp_manager
            }
            */
            auto create_manager_begin = std::chrono::steady_clock::now();
            std::string name = "thread-" + std::to_string(thread_id) + "-stream-" + stream_name;
            decomp_nvcomp_manager = std::make_shared<SyncedGdeflateManager>(1 << 16, nvcompBatchedGdeflateDefaultOpts, stream, name); // 1 << 16 is 64KB, 0 is fast compression
            // 1 << 16 is 64KB, 0 is fast compression
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

          debug(prefix + "decompressed in " + pprint(decomp_time) + ", freeing with stream " + stream_name + "");
          if (!REUSE_COMP_BUFFER) {
            // need to not free comp_buffer until the decompression is complete, so we should 
            CUDA_CHECK(cudaFreeAsync(comp_buffer, stream));
            if (SYNC_FREE)
              CUDA_CHECK(cudaStreamSynchronize(stream));
          }
          thread_decomp_time += decomp_time; // shrug
        }

        ms_t file_elapsed_time = std::chrono::duration_cast<ms_t>(std::chrono::steady_clock::now() - file_start_time);
        // ideally in debug mode this would mention copy and decomp time 
        log(prefix + "processed file in " + pprint(file_elapsed_time) + " (" + pprint_throughput(files[i].decompressed_size, file_elapsed_time) + ")");
        thread_copy_time += copy_time;
      }
      if (REUSE_COMP_BUFFER) {
        cudaStream_t stream = streams[thread_id][0];
        log("thread " + std::to_string(thread_id) + "freeing with stream " + pprint(stream) + "");
        // need to not free comp_buffer until the decompression is complete, so we should 
        CUDA_CHECK(cudaFreeAsync(saved_comp_buffer, stream));
        if (SYNC_FREE)
          CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      auto thread_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - thread_start);
      auto thread_elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(thread_elapsed);

      log("thread " + std::to_string(thread_id) + " processed " + pprint(thread_decompressed_size) + " in " + pprint(thread_elapsed) + " (" + pprint_throughput(thread_decompressed_size, thread_elapsed) + " ) - " + std::to_string(indexes.size()) + " files");
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
    } else {
      std::cerr << "a copy_done event is null, a thread must have failed to start or not done any work" << std::endl;
    }
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

  size_t total_decompressed_size = 0;
  for (auto& file : files)
    total_decompressed_size += file.decompressed_size;

  auto end_time = std::chrono::steady_clock::now();
  auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  log("Processing throughput: " + pprint_throughput(total_decompressed_size, end_time - start_time));
  log("Total processing time: " + std::to_string(elapsed_time) + " ms for " + std::to_string(num_files) + " tensors on " + std::to_string(num_threads) + " threads, " + std::to_string(num_streams) + " streams");
  log("Total copy time: " + std::to_string(total_copy_time.count()) + "[ms], total decomp time: " + std::to_string(total_decomp_time.count()) + "[ms], total read time: " + std::to_string(total_read_time.count()) + "[ms]");
  log("Average copy time per file: " + std::to_string((total_copy_time / num_files).count()) + "[ms], average decomp time per file: " + std::to_string((total_decomp_time / num_files).count()) + "[ms]");
  log("Average copy time per thread: " + std::to_string((total_copy_time / num_threads).count()) + "[ms], average decomp time per thread: " + std::to_string((total_decomp_time / num_threads).count()) + "[ms]");

  std::ofstream file("/tmp/stats.json", std::ios::app);
  file << "{\"elapsed_time\":" << elapsed_time << ",\"num_files\":" << num_files;
  file << ",\"num_threads\":" << num_threads << ",\"num_streams\":" << num_streams;
  file << ",\"total_copy_time\":" << total_copy_time.count() << ",\"total_decomp_time\":" << total_decomp_time.count();
  file << ",\"total_file_size\":" << total_decompressed_size << ",\"chunk_size\":" << CHUNK_SIZE;
  file << ",\"sleep\":" << getenv("SLEEP", 3) << ",\"total_read_time\":" << total_read_time.count();
  file << ", \"name\":\"" << (std::getenv("NAME") ? std::getenv("NAME") : "unknown") << "\"";
  file << "}" << std::endl;
  file.close();

  return tensors;
}


std::vector<torch::Tensor> decompress_from_meta(const std::string& fname) {
  return std::apply(batch_decompress, load_any_csv(fname));
}

class AsyncDecompressor {
 private:
  std::future<std::vector<torch::Tensor>> future;

 public:
  bool done = false;
  bool failed = false;
  std::string error;

  explicit AsyncDecompressor(std::string fname) {
    std::vector<CompressedFile> files;
    std::vector<std::vector<int>> thread_to_idx;
    try {
      std::tie(files, thread_to_idx) = load_any_csv(fname);
    }
    catch(const std::exception& e) {
      std::cerr << e.what() << '\n';
      failed = true;
    }
    if (failed) return;
    log("starting batch_decompress future");
    future = std::async(std::launch::async, batch_decompress, files, thread_to_idx);
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

  m.def("compress", &compress, R"pbdoc(compress bytes to a file. algo is 0/1/2 default/fast compression/best compression.)pbdoc", py::arg("data"), py::arg("filename"), py::arg("algo") = 0);

  m.def("decompress", &decompress, "decompress to a new tensor", py::arg("filename"), py::arg("shape"), py::arg("dtype"));

  m.def("batch_decompress", &batch_decompress, "decompress tensors with a threadpool", py::arg("files"), py::arg("assignments"), py::call_guard<py::gil_scoped_release>());

  m.def("decompress_from_meta", &decompress_from_meta, "decompress tensors from meta.csv", py::arg("filename"), py::call_guard<py::gil_scoped_release>());

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
