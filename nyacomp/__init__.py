import contextlib
import ctypes
import os
import pickle
import time
from typing import Iterator
import numpy as np
import concurrent

# import pycuda.autoinit
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


def compress(
    model: torch.nn.Module | dict, name: str = "model.pth", threshold: int = 16000
) -> float:
    if isinstance(model, torch.nn.Module):
        parameters = list(model.parameters())
    else:
        parameters = [value for key, value in sorted(model.items())]
    total_size = 0.0
    total_compressed_size = 0
    meta = []
    for i, param in enumerate(parameters):
        data = tensor_bytes(param.data.detach().cpu())
        total_size += float(len(data))
        if len(data) >= threshold:  # 4 KB
            print(f"compressing parameter {i}")
            total_compressed_size += _nyacomp.compress(data, f"tensors/{i}.gz")
            meta.append({"shape": param.shape, "dtype": param.dtype})
            param.data = torch.tensor([], dtype=param.dtype)
        else:
            total_compressed_size += len(data)
            meta.append({})
    pickle.dump(meta, open("tensors/metadata.pkl", "wb"))
    print("overall tensor size: ", total_size)
    print("overall compression ratio:", total_compressed_size / total_size)
    torch.save(model, name)
    return total_compressed_size / total_size


def load_compressed(fname: str = "model.pth") -> torch.nn.Module | dict:
    model = torch.load(fname)
    metadata = pickle.load(open("tensors/metadata.pkl", "rb"))
    if isinstance(model, torch.nn.Module):
        params = list(model.parameters())
    else:
        params = [value for key, value in sorted(model.items())]
    for i, (param, meta) in enumerate(zip(params, metadata)):
        if meta:
            param.data = torch.empty(
                meta["shape"], dtype=meta["dtype"], device="cuda:0"
            )
            param.data = _nyacomp.decompress(f"tensors/{i}.gz", param.data)
        # print(param.data.shape)
        # assert param.data.abs().sum().item() != 0.0
    return model


def load_compressed_threaded(fname: str = "model.pth") -> torch.nn.Module | dict:
    model = torch.load(fname)
    metadata = pickle.load(open("tensors/metadata.pkl", "rb"))
    if isinstance(model, torch.nn.Module):
        params = list(model.parameters())
    else:
        params = [value for key, value in sorted(model.items())]

    for param, meta in zip(params, metadata):
        if meta:
            param.data = torch.empty(
                meta["shape"], dtype=meta["dtype"], device="cuda:0"
            )
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(_nyacomp.decompress, f"tensors/{i}.gz", param.data)
            if meta
            else None
            for i, (param, meta) in enumerate(zip(params, metadata))
        ]
        for param, future in zip(params, futures):
            if future:
                param.data = future.result()
    return model
