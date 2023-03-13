import contextlib
import ctypes
import os
import pickle
import time
from typing import Iterator
import numpy as np
import pycuda.autoinit
import torch
import _nyacomp

@contextlib.contextmanager
def timer(msg: str) -> Iterator[None]:
    start = time.time()
    yield
    print(f"{msg} took {time.time() - start:.3f}s")


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    length = int(np.prod(tensor.shape).item())
    bytes_per_item = torch._utils._element_size(tensor.dtype)

    total_bytes = length * bytes_per_item

    ptr = tensor.data_ptr()
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy
    return data.tobytes()


def compress(model: torch.nn.Module, name: str = "model.pth") -> float:
    parameters = list(model.parameters())
    total_size = 0.0
    total_compressed_size = 0
    for i, param in enumerate(parameters):
        data = tensor_bytes(param.data.detach().cpu())
        total_size += float(len(data))
        print(f"compressing parameter {i}")
        total_compressed_size += _nyacomp.compress(data, f"tensors/{i}.gz")

    meta = [{"shape": param.shape, "dtype": param.dtype} for param in parameters]
    pickle.dump(meta, open("tensors/metadata.pkl", "wb"))

    for param in parameters:
        param.data = torch.tensor([], dtype=param.dtype)
    print("overall tensor size: ", total_size)
    print("overall compression ratio:", total_compressed_size / total_size)
    torch.save(model, name)
    return total_compressed_size / total_size


def load_compressed(fname: str = "model.pth") -> torch.nn.Module:
    model = torch.load(fname)
    metadata = pickle.load(open("tensors/metadata.pkl", "rb"))
    params = list(model.parameters())
    for i, param in enumerate(params):
        param.data = torch.empty(
            metadata[i]["shape"], dtype=metadata[i]["dtype"], device="cuda:0"
        )
        param.data = _nyacomp.decompress(f"tensors/{i}.gz", param.data)
        # print(param.data.shape)
        # assert param.data.abs().sum().item() != 0.0
    return model
