import contextlib
import ctypes
import os
import pickle
import time
from typing import Iterator
import numpy as np
import pycuda.autoinit
import torch
import python_example as nvcomp

@contextlib.contextmanager
def timer(msg: str) -> Iterator[None]:
    start = time.time()
    yield
    print(f"{msg} took {time.time() - start:.3f}s")


# torch._utils._element_size(dtype)
_SIZE = {
    torch.int64: 8,
    torch.float32: 4,
    torch.int32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int16: 2,
    torch.uint8: 1,
    torch.int8: 1,
    torch.bool: 1,
    torch.float64: 8,
}


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    length = int(np.prod(tensor.shape).item())
    bytes_per_item = torch._utils._element_size(tensor.dtype)

    total_bytes = length * bytes_per_item

    ptr = tensor.data_ptr()
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    # print("getting tensor bytes")
    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy
    return data.tobytes()


def compress(model: torch.nn.Module, name: str = "model.pth") -> float:
    print("a")
    parameters = list(model.parameters())
    total_size = 0.0
    total_compressed_size = 0
    print("b")
    for i, param in enumerate(parameters):
        data = tensor_bytes(param.data.detach().cpu())
        total_size += float(len(data))
        print(f"compressing parameter {i}")
        total_compressed_size += nvcomp.compress(data, f"tensors/{i}.gz")

    meta = [{"shape": param.shape, "dtype": param.dtype} for param in parameters]
    pickle.dump(meta, open("tensors/metadata.pkl", "wb"))

    for param in parameters:
        param.data = torch.tensor([], dtype=param.dtype)
    print("overall tensor size: ", total_size)
    print("overall compression ratio:", total_compressed_size / total_size)
    torch.save(model, name)
    print(os.stat(name).st_size)
    return total_compressed_size / total_size


def load_compressed(fname: str = "model.pth") -> torch.nn.Module:
    model = torch.load(fname)
    metadata = pickle.load(open("tensors/metadata.pkl", "rb"))
    params = list(model.parameters())
    for i, param in enumerate(params):
        param.data = torch.empty(
            metadata[i]["shape"], dtype=metadata[i]["dtype"], device="cuda:0"
        )
        param.data = nvcomp.decompress(f"tensors/{i}.gz", param.data)
        print(param.data.shape)
        # assert param.data.abs().sum().item() != 0.0
    return model

if False:
    model = torch.load(
        "/home/sylv/dryad/sprkpnt/vqgan/predict/reaction_predictor_no_gauss.pth",
        map_location="cpu",
    ).eval()
    print("loaded model")
    ratio = compress(model)
    print(f"gdeflate compression ratio: {ratio:.4f}")

with timer("loading with nvcomp"):
    new_model = load_compressed().eval()

with timer("torch.load to cpu"):
    model = torch.load(
        "/home/sylv/dryad/sprkpnt/vqgan/predict/reaction_predictor_no_gauss.pth",
        map_location="cpu",
    ).eval()
with timer(".cuda()"):
    model = model.cuda()

# test_input = torch.zeros([768], device="cuda:0")
# print("zeros:", new_model(test_input).eq(model(test_input)))
# test_input = torch.ones([768], device="cuda:0")
# print("ones:", new_model(test_input).eq(model(test_input)))

rainbow = torch.load("./rainbow_embed.pth").to("cuda:0").float()
print('decompressed reaction predictor score for "rainbow":', new_model(rainbow))
print('original reaction predictor score for "rainbow":', model(rainbow))

# print((new_model(rainbow) == model(rainbow)).all())
