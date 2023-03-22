import asyncio

import contextlib
import ctypes
import os
import pickle
import time
from typing import Iterator

# import numpy as np
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


def tensor_bytes(tensor: "torch.Tensor") -> bytes:
    import torch
    import numpy as np

    length = int(np.prod(tensor.shape).item())
    bytes_per_item = torch._utils._element_size(tensor.dtype)

    total_bytes = length * bytes_per_item

    ptr = tensor.data_ptr()
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy
    return data.tobytes()


def compress_model(model: str | None = None) -> None:
    if model:
        components = (
            Path(f"~/.cache/huggingface/diffusers/{model}/snapshots")
            .expanduser()
            .glob("**/*bin")
        )
    else:
        components = (
            Path("~/.cache/huggingface/diffusers/")
            .expanduser()
            .glob("*/snapshots/**/*bin")
        )
    compress_components(components)


def compress_components(components: list[Path]) -> None:
    total_size, total_compressed_size = 0, 0
    for component in list(components):
        if "nya" in str(component) or "boneless" in str(component):
            continue
        size, comp_size = compress_state_dict(component)
        total_size += size
        total_compressed_size += comp_size
    print("all components ratio:", total_compressed_size / total_size)


def compress_state_dict(_path: str, treshold: int = 16000) -> tuple[int, int]:
    import torch

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
            new_size = _nyacomp.compress(data, f"{dir / key}.gz")
            if new_size > len(data):
                print("bad compression ratio despite heuristic, deleting ", key)
                (dir / f"{key}.gz").unlink()
            else:
                total_compressed_size += new_size
                state_dict[key] = {
                    "shape": value.shape,
                    "dtype": value.dtype,
                    "len": len(data),
                }
                # param.data = torch.tensor([], dtype=param.dtype)
        else:
            total_compressed_size += len(data)
    print("overall tensor size: ", total_size)
    print("overall compression ratio:", total_compressed_size / float(total_size))
    torch.save(state_dict, dir / f"boneless_{path.name}")
    return (total_size, total_compressed_size)


# def batch_load_compressed_state_dict(path: str = "model.pth", fn="_async") -> dict:
#     dir = Path(path).parent / "nya"
#     state_dict = torch.load(dir / f"boneless_{Path(path).name}")
#     with timer("allocate empties"):
#         pairs = [
#             (
#                 key,
#                 f"{dir / key}.gz",
#                 torch.empty(value["shape"], dtype=value["dtype"], device="cuda:0"),
#             )
#             for key, value in state_dict.items()
#             if isinstance(value, dict)
#         ]
#         keys, fnames, tensors = map(list, zip(*pairs))
#     with timer("decompress"):
#         getattr(_nyacomp, f"decompress_batch{fn}")(fnames, tensors)
#         # if _async:
#         #     _nyacomp.decompress_batch_async(fnames, tensors)
#         # else:
#         #     _nyacomp.decompress_batch(fnames, tensors)

#     state_dict = state_dict | {key: tensor for key, tensor in zip(keys, tensors)}
#     # copy, decomp = map(sum, zip(*[future.result() for future in futures]))
#     # print(f"total copy time: {copy/1000}ms, total decomp time: {decomp/1000}ms")
#     return state_dict


def dry_load(path: str) -> dict:
    dir = Path(path).parent / "nya"
    state_dict = torch.load(dir / f"boneless_{Path(path).name}")

    keys = [k for k, v in state_dict.items() if isinstance(v, dict)]
    fnames = [f"{dir / key}.gz" for key in keys]
    shapes = [list(state_dict[k]["shape"]) for k in keys]
    dtypes = [str(state_dict[k]["dtype"]).split(".")[1] for k in keys]
    json.dump([fnames, shapes, dtypes], open("/tmp/shapes", "w"))


# search partitioning space in advance to find optimal thread packing
# we want tensors of the same size to be together while making sure each thread has similar amounts of work
# we can do this by sorting the tensors by size and then assigning them to threads in order
# when we can we also want to assign work to the thread with the least amount of work

# find the bin packing with the shortest max bin size
# then, find the bin packing with the highest number of repeated sizes with the same max bin size

# if the the entire slowest thread is one tensor and cannot be packed differently,
# all the other bins can use that to pack tensors of the same size


def score(binning: list[list[int]]) -> tuple[int, int]:
    longest = max(map(sum, binning))
    changes = 0
    for bin in binning:
        for prev_size, next_size in zip(bin[:-1], bin[1:]):
            if prev_size != next_size:
                changes += 1
    changes = sum(
        1
        for prev_size, next_size in zip(bin[:-1], bin[1:])
        for bin in binning
        if prev_size != next_size
    )



def partition(sizes: list[int], bins: int = 32) -> list[list[int]]:
    indexes = [[] for _ in range(bins)]
    bin_sizes = [0] * bins

    lemgthy = max(sizes)

    initial = sorted(enumerate(sizes), key=lambda x: x[1], reverse=True)
    for i, size in initial:
        smallest = bin_sizes.index(min(bin_sizes))
        bin_sizes[smallest] += size
        indexes[smallest].append(i)

    while 1:
        smallest, biggest = map(bin_sizes.index, (min(bin_sizes), max(bin_sizes)))
        diff = bin_sizes[biggest] - bin_sizes[smallest]
        _, index_to_move = max(
            [(sizes[i], i) for i in indexes[biggest] if sizes[i] * 2 < diff],
            default=(None, None),
        )
        if not index_to_move:
            break
        indexes[biggest].remove(index_to_move)
        bin_sizes[biggest] -= sizes[index_to_move]
        indexes[smallest].append(index_to_move)
        bin_sizes[smallest] += sizes[index_to_move]

    def indexes_to_sizes(indx: list[list[int]]) -> list[list[int]]:
        return [[sizes[i] for i in bin] for bin in indx]

    counts = [Counter([sizes[i] for i in bin]) for bin in indx]
    size_bins = indexes_to_sizes(indexes)

    for bin_number, bin in enumerate(bin_number, size_bins):
        for i, item in enumerate(bin):
            for replacement_bin in size_bins:
                if replacement_bin[item] > bin[item]:
                    pass # swap


    return indexes


def good_load(path: str) -> dict:
    dir = Path(path).parent / "nya"
    state_dict = torch.load(dir / f"boneless_{Path(path).name}")

    keys = [k for k, v in state_dict.items() if isinstance(v, dict)]
    fnames = [f"{dir / key}.gz" for key in keys]
    shapes = [list(state_dict[k]["shape"]) for k in keys]
    dtypes = [str(state_dict[k]["dtype"]).split(".")[1] for k in keys]
    sizes = [state_dict[k]["len"] for k in keys]

    # tensors = _nyacomp.good_batch_decompress_threadpool(fnames, shapes, dtypes, -1, -1)
    tensors = _nyacomp.batch_decompress_threadpool(fnames, shapes, dtypes)
    if None in tensors:
        import pdb

        pdb.set_trace()
    # tensors = _nyacomp.decompress_batch_async_new(fnames, shapes, dtypes)
    return state_dict | dict(zip(keys, tensors))


def toggle_patch():
    import diffusers

    if diffusers.modeling_utils.load_state_dict == good_load:
        diffusers.modeling_utils.load_state_dict = (
            diffusers.modeling_utils._load_state_dict
        )
    else:
        diffusers.modeling_utils._load_state_dict = (
            diffusers.modeling_utils.load_state_dict
        )
        diffusers.modeling_utils.load_state_dict = good_load


def stats(times):
    import statistics

    stats = {
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times),
        "min": min(times),
    }
    return " ".join(f"{k}: {round(v, 4)}" for k, v in stats.items())


# toggle_patch()
# import nyacomp
try:
    guy = str(
        list(
            Path("~/.cache/huggingface/hub")
            .expanduser()
            .glob("models--o*/snapshots/*/*bin")
        )[0]
    )
    # print(guy)
except IndexError:
    pass
import timeit

# compress_state_dict(str(guy))
if __name__ == "__main__":
    # torch.cuda.synchronize()

    times = [
        timeit.timeit("good_load(guy)", number=1, globals=globals()) for i in range(4)
    ]
    print("good_load: ", stats(times))
    t_res = timeit.timeit(
        "torch.load(guy, map_location='cuda:0')", number=2, globals=globals()
    )
    print("torch: ", t_res / 2)
    # dd=good_load(guy)
    # with nyacomp.timer("good:"):    dd=good_load(guy)
    # dd=asyncio.run(lazy_load(guy))#, "_threaded")
    # with nyacomp.timer("torch:"):        dd_t = torch.load(guy, map_location="cuda:0")
    # with nyacomp.timer("cuda:"):
    #     for v in dd_t.values():
    #         v.cuda()

    # with nyacomp.timer("load_compressed"):
    #     model = diffusers.StableDiffusionPipeline.from_pretrained(
    #         torch_dtype=torch.float16,
    #         revision="fp16",
    #         safety_checker=None,
    #         pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
    #         local_files_only=True,
    #     )
    # with nyacomp.timer("cuda"):
    #     model.cuda()
