import python_example
import torch
import time
import pickle

# array = np.ones((100,), dtype=np.uint8)
# gpu_array = pycuda.gpuarray.to_gpu(array)
# size = array.size * array.itemsize

original_tensor = torch.ones(2**20, dtype=torch.uint8)


filename = "input.bin"
open(filename, "wb").write(b"\x01" * 2**20)
print("=compressing=")
python_example.compress(filename)

print("=decompressing=")
comp_start = time.time()
decompressed_tensor = torch.empty(2**20, dtype=torch.uint8, device="cuda:0")
python_example.decompress("compressed.bin", decompressed_tensor)
print(f"loading with gpu decompression took {time.time() - comp_start:.4f}s")
torch.save(original_tensor, "/tmp/ones.pth")
unpickle_start = time.time()
unpickled = torch.load("/tmp/ones.pth")
print(f"torch.load took {time.time() - unpickle_start:.4f}s")
copy_start = time.time()
unpickled.cuda()
print(f".cuda() took {time.time() - copy_start:.4f}s")
print("original_tensor.cuda().eq(decompressed_tensor).all(): ", original_tensor.cuda().eq(decompressed_tensor).all())
#print(result)
#print(python_example.make_tensor()
