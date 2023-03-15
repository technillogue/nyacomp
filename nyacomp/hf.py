import contextlib
import ctypes
import os
import pickle
import time
from typing import Iterator
import numpy as np
import concurrent
from pathlib import Path

# import pycuda.autoinit
import torch
import _nyacomp
import diffusers


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


def compress_model():
    components = (
        Path("~/.cache/huggingface/diffusers/").expanduser().glob("*/snapshots/**/*bin")
    )
    total_size, total_compressed_size = 0, 0
    for component in list(components):
        if "nya" in str(component) or "boneless" in str(component):
            continue
        size, comp_size = compress_state_dict(component)
        total_size += size
        total_compressed_size += comp_size
    print("all components ratio:", total_compressed_size / total_size)


def compress_state_dict(_path: str, treshold: int = 16000) -> (int, int):
    path = Path(_path)
    state_dict = torch.load(path)

    total_size = 0
    total_compressed_size = 0
    dir = path.parent / "nya"
    dir.mkdir(exist_ok=True)

    for key, value in state_dict.items():
        data = tensor_bytes(value.detach().cpu())  # gah
        total_size += len(data)
        if len(data) > treshold:
            print(f"compressing parameter {key}")
            total_compressed_size += _nyacomp.compress(data, f"{dir / key}.gz")
            state_dict[key] = {"shape": value.shape, "dtype": value.dtype}
            # param.data = torch.tensor([], dtype=param.dtype)
        else:
            total_compressed_size += len(data)
    print("overall tensor size: ", total_size)
    print("overall compression ratio:", total_compressed_size / float(total_size))
    torch.save(state_dict, dir / f"boneless_{path.name}")
    return (total_size, total_compressed_size)


def load_compressed_state_dict(path: str = "model.pth") -> dict:
    dir = Path(path).parent / "nya"
    state_dict = torch.load(dir / f"boneless_{Path(path).name}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for key, value in state_dict.items():
            if isinstance(value, dict):  # maybe check for tensor specifically?
                print("decompressing")
                state_dict[key] = torch.empty(
                    value["shape"], dtype=value["dtype"], device="cuda:0"
                )
                executor.submit(_nyacomp.decompress, f"{dir / key}.gz", state_dict[key])
            else:
                print("not decompressing", key)
    print("exited pool")
    return state_dict


diffusers.modeling_utils._load_state_dict = diffusers.modeling_utils.load_state_dict
diffusers.modeling_utils.load_state_dict = load_compressed_state_dict
import nyacomp

with nyacomp.timer("awa"):
    model = diffusers.StableDiffusionPipeline.from_pretrained(
        torch_dtype=torch.float16,
        revision="fp16",
        safety_checker=None,
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
        local_files_only=True,
    )
