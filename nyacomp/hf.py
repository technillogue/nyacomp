import asyncio
import contextlib
import ctypes
import os
import time
from pathlib import Path
from typing import Iterator

import _nyacomp
# import pycuda.autoinit
import torch

#import diffusers
#from nyacomp import partition
import partition

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
                    "len_compressed": new_size,
                }
                # param.data = torch.tensor([], dtype=param.dtype)
        else:
            total_compressed_size += len(data)
    #state_dict["meta"] = massage(
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


def good_load(path: str) -> dict:
    dir = Path(path).parent / "nya"
    state_dict = torch.load(dir / f"boneless_{Path(path).name}")

    keys = [k for k, v in state_dict.items() if isinstance(v, dict)]
    keys.sort(key=lambda k:state_dict[k]["len"], reverse=True)
    files = [
        _nyacomp.CompressedFile(
            f"{dir / k}.gz",
            list(state_dict[k]["shape"]),
            str(state_dict[k]["dtype"]).removeprefix("torch."),
            state_dict[k]["len"],
        )
        for k in keys
    ]
    threads = int(os.getenv("NUM_THREADS", os.cpu_count()))
    assignments = partition.massage(tuple(state_dict[k]["len"] for k in keys), threads)
    for bin in assignments:
        bin.sort(key=lambda k:state_dict[keys[k]]["len_compressed"], reverse=True)

    size = str(sum(state_dict[keys[bin[0]]]["len_compressed"] for bin in assignments))
    os.environ["TOTAL_FILE_SIZE"] = size

    # tensors = _nyacomp.good_batch_decompress_threadpool(fnames, shapes, dtypes, -1, -1)
    tensors = _nyacomp.batch_decompress_threadpool(files, assignments)
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

#compress_state_dict(str(guy))
if __name__ == "__main__":
    torch.cuda.synchronize()

    times = [
        timeit.timeit("good_load(guy)", number=1, globals=globals()) for i in range(4)
    ]
    print("good_load: ", stats(times))
    t_res = timeit.timeit(
        "torch.load(guy, map_location='cuda:0')", number=2, globals=globals()
    )
    print("torch: ", t_res / 2)
    #dd = good_load(guy)
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
