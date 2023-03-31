
size_t pow2_ceil(size_t size) {  
  // return smallest power of two that is greater than size
  return 1 << (size_t)(log2(size - 1) + 1);
}
size_t exp(size_t size) {
  return (size_t)(log2(size - 1) + 1);
}

class Pool {
  public:
    Pool(std::vector<size_t> initial_slabs, cudaStream_t stream) :
      stream(stream), shared_buffers(1), running(true) {

      std::vector<size_t> initial_sizes_round;
      for (size_t size : initial_slabs) {
        initial_sizes_round.push_back(exp(size));
      }
      
      smallest = (int)*std::min_element(initial_sizes_round.begin(), initial_sizes_round.end());
      auto max = *std::max_element(initial_sizes_round.begin(), initial_sizes_round.end());
      slab_indexes_by_size.resize(max - smallest + 1);

      size_t total_initial_size = std::accumulate(initial_sizes_round.begin(), initial_sizes_round.end(), 0) * 1.5; // maybe add some padding here
      max_slabs = initial_slabs.size() + 500; // actually files.size()
      size_t release_count_tracking_size = max_slabs * sizeof(uint);
      
      shared_buffer_sizes.push_back(release_count_tracking_size + total_initial_size);
      CUDA_CHECK(cudaMallocAsync(&shared_buffers[0], release_count_tracking_size + total_initial_size, stream));
      // set the release_count part of the buffer to (uint)0
      CUDA_CHECK(cudaMemsetAsync(shared_buffers[0], 0, release_count_tracking_size, stream));
      // make device_release_counts point at that first part of shared_buffers[0], but use it as uint32_t*
      device_release_counts = (uint32_t*)shared_buffers[0];
      
      shared_offsets.push_back(std::atomic<size_t>(release_count_tracking_size));

      CUDA_CHECK(cudaMallocHost(&cached_release_counts, release_count_tracking_size));
      // set host cached_release_counts to 0
      memset(cached_release_counts, 0, release_count_tracking_size);

      // now acquire_counts, cached_release_counts and device_release_counts are both 0
      // and shared_buffers[0] is allocated and shared_offsets[0] is set to the end of the release_count part
      // so let's slice up the initial allocations

      for (size_t i = 0; i < initial_sizes_round.size(); i++) {
        int size_exp = exp(initial_sizes_round[i]);
        slab_indexes_by_size[size_exp - smallest].push_back(i);
        // slab_sizes.push_back(initial_sizes_round[i]); // unclear if needed
        slabs.push_back(shared_buffers[0] + shared_offsets[0].fetch_add(initial_sizes_round[i]));
        acquire_counts.push_back(0);
      };
    }

    ~Pool() {
      // check_releases_future.cancel();
      CUDA_CHECK(cudaFreeHost(cached_release_counts));
      for (auto shared_buffer : shared_buffers) {
        CUDA_CHECK(cudaFree(shared_buffer));
      }

    }

    std::pair<size_t, uint8_t*> acquire(size_t size) {
      if (check_releases_future == nullptr)
        check_releases_future = std::async(std::launch::async, &Pool::check_releases_task, this);
      int size_idx = exp(size) - smallest;
      // iterate over teh lists of slabs of size size_exp or greater
      for (int i = size_idx; i < slab_indexes_by_size.size(); i++) {
        for (size_t slab_id : slab_indexes_by_size[i]) {
          // if the slab is not in use
          if (acquire_counts[slab_id] == cached_release_counts[slab_id]) {
            // increment the acquire count
            acquire_counts[slab_id]++;
            // return the slab
            return std::make_pair(slab_id, slabs[slab_id]);
          }
          if (acquire_counts[slab_id] < cached_release_counts[slab_id]) {
            throw std::runtime_error("slab has been released more times than acquired!!");
          }
        }
      };
      if (max_slabs == slabs.size()) {
        throw std::runtime_error("Pool is full");
      }
      // if we get here, we need to allocate a new slab
      size_t new_size = pow2_ceil(size);
      if (shared_buffer_sizes.back() < shared_offsets.back().load() + new_size) {
        // if there is not enough space in the current buffer, allocate a new buffer
        // ideally free or reuse the remainder of the old buffer, like maybe a new slab could be allocated from the end of the old buffer
        size_t new_shared_size = std::max(new_size, (size_t)(shared_buffer_sizes.back() * 1.5));
        shared_buffer_sizes.push_back(new_shared_size);
        shared_buffers.push_back(nullptr);
        CUDA_CHECK(cudaMallocAsync(&shared_buffers.back(), new_size, stream));
      }

      // allocate a new slab
      // if slab_indexes_by_size doesn't have the right exponent yet, expand it
      if (slab_indexes_by_size.size() <= exp(new_size) - smallest) {
        slab_indexes_by_size.resize(exp(new_size) - smallest + 1);
      }
      slab_indexes_by_size[exp(new_size) - smallest].push_back(slabs.size());
      slabs.push_back(shared_buffers.back() + shared_offsets.back().fetch_add(new_size));
      // increment the acquire count
      acquire_counts.push_back(1);
      return std::make_pair(slabs.size() - 1, slabs.back());
    }

    void check_releases_task() {
      while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        CUDA_CHECK(cudaMemcpyAsync(cached_release_counts, device_release_counts, max_slabs * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
      }
    }
  
  private:
    size_t max_slabs;
    std::vector<uint32_t> acquire_counts;
    uint32_t* device_release_counts;
    uint32_t* cached_release_counts;
    std::future<void> check_releases_future;

    std::vector<uint8_t*> shared_buffers;
    std::vector<size_t> shared_buffer_sizes;
    std::vector<std::atomic<size_t>> shared_offsets;

    bool running;
    
    int smallest;    
    std::vector<std::vector<size_t>> slab_indexes_by_size;
    std::vector<uint8_t*> slabs;

    cudaStream_t stream;
};



/*
// cuda_kernels.cu
#include <cuda_runtime.h>

__global__ void increment_release_count(uint* device_release_counts, int index) {
    atomicAdd(device_release_counts + index, 1);
}

extern "C" void increment_release_count_kernel(uint* device_release_counts, int index, cudaStream_t stream) {
    increment_release_count<<<1, 1, 0, stream>>>(device_release_counts, index);
}
*/

extern "C" void increment_release_count_kernel(uint* device_release_counts, int index, cudaStream_t stream);


class RefCounter {
public:
  RefCounter(size_t size) : size(size), acquire_counts(size, 0), should_sync(true) {
    size_t tracking_size = size * sizeof(uint);
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMallocAsync(&device_release_counts, tracking_size, stream));
    CUDA_CHECK(cudaMemsetAsync(device_release_counts, 0, tracking_size, stream));
    CUDA_CHECK(cudaMallocHost(&cached_release_counts, tracking_size));
    memset(cached_release_counts, 0, tracking_size);
  }

  ~RefCounter() {
    should_sync = false;
    CUDA_CHECK(cudaFreeHost(cached_release_counts));
    CUDA_CHECK(cudaFree(device_release_counts));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void acquire(size_t index) {
    acquire_counts[index]++;
    if (monitor_future == nullptr || monitor_future.valid() == false)
      monitor_future = std::async(std::launch::async, &RefCounter::monitor, this);
  }

  void release(size_t index, cudaStream_t release_stream) {
    increment_release_count_kernel(device_release_counts, index, release_stream);
  }

  void monitor() {
    while (should_sync) {
      CUDA_CHECK(cudaMemcpyAsync(cached_release_counts, device_release_counts, size * sizeof(uint), cudaMemcpyDeviceToHost, stream));
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }

  bool is_free(size_t index) {
    return acquire_counts[index] == cached_release_counts[index];
  }

private:
  size_t size;
  std::vector<uint32_t> acquire_counts;
  uint32_t* device_release_counts;
  uint32_t* cached_release_counts;
  std::future<void> monitor_future;
  cudaStream_t stream;
  bool should_sync;
};


