import os
import sys
import importlib.abc
import importlib.util

if os.getenv("HIDE_MODULES"):
    hidden = {"jax", "flax", "accelerate", "wandb"}

    def find_spec(name, package=None):
        if name in hidden:
            return None
        return orig_find_spec(name, package)

    class CustomFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname in hidden:
                raise ImportError(f"{fullname} is blocked and cannot be imported.")
            return None

    orig_find_spec = importlib.util.find_spec
    importlib.util.find_spec = find_spec
    sys.meta_path.insert(0, CustomFinder())

import contextlib
import ctypes
import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import partition
from nvtx import annotate

import _nyacomp


@contextlib.contextmanager
def timer(msg: str) -> "Iterator[None]":
    start = time.time()
    yield
    print(f"{msg} took {time.time() - start:.3f}s")


with timer("import torch"):
    import torch


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    length = int(np.prod(tensor.shape).item())
    bytes_per_item = torch._utils._element_size(tensor.dtype)

    total_bytes = length * bytes_per_item

    ptr = tensor.data_ptr()
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy
    return data.tobytes()


Tensory = torch.Tensor | torch.nn.Parameter


def compress_parameter(param: Tensory, path: Path) -> tuple[dict, int, int]:
    data = tensor_bytes(param.data.detach().cpu())
    size = len(data)
    if len(data) >= 1 << 14:  # 4 KB
        if path.exists() and not os.getenv("RECOMPRESS"):
            print(f"skipping already existing parameter at {path}")
            new_size = path.stat().st_size
        else:
            print(f"compressing parameter to {path}")
            new_size = _nyacomp.compress(data, str(path))
        if new_size < size:
            meta = {
                "filename": str(path),
                "shape": list(param.shape),
                "dtype": str(param.dtype).removeprefix("torch."),
                "decompressed_size": size,
                "compressed_size": new_size,
            }
            # torch.nn.parameter.UninitializedParameter
            param.data = torch.tensor([], dtype=param.dtype)
            return meta, size, new_size
        print("bad compression ratio despite heuristic, deleting ", path)
        path.unlink()
    return {}, size, size


default_path = Path("./boneless_model.pth")


def compress(model: torch.nn.Module | dict, path: Path = default_path) -> float:
    if isinstance(model, torch.nn.Module):
        # parameters = list(model.named_parameters()))
        parameters = list(model.parameters())
    else:
        parameters = list(sorted(model.items()))
    dir = path.parent / "nya"
    dir.mkdir(exist_ok=True)

    total_size = 0.0
    total_compressed_size = 0
    meta = []

    for i, param in enumerate(parameters):
        param_path = dir / f"{i}.gz"
        param_meta, size, new_size = compress_parameter(param, param_path)
        meta.append(param_meta)
        total_size += size
        total_compressed_size += new_size

    threads = int(os.getenv("NUM_THREADS") or os.cpu_count())
    sizes = tuple(param_meta["compressed_size"] for param_meta in meta if param_meta)
    assignments = partition.massage(sizes, threads)

    for bin in assignments:
        biggest_file = max(bin, key=lambda idx: sizes[idx])
        bin.remove(biggest_file)
        bin.insert(0, biggest_file)

    meta.append(assignments)

    pickle.dump(meta, open(str(dir / "metadata.pkl"), "wb"))
    print("overall compression ratio:", total_compressed_size / total_size)
    print("saving boneless model to ", path)
    import pdb

    torch.save(model, str(path))
    pdb.set_trace()
    return total_compressed_size / total_size


@annotate("load_compressed")
def load_compressed(path: Path = default_path) -> torch.nn.Module | dict:
    dir = path.absolute().parent / "nya"

    metadata = pickle.load(open(dir / "metadata.pkl", "rb"))
    assignments = metadata.pop()

    real_meta = [meta for meta in metadata if meta]

    files = [
        _nyacomp.CompressedFile(
            meta["filename"],
            meta["shape"],
            meta["dtype"],
            meta["decompressed_size"],
        )
        for meta in real_meta
    ]

    threads = int(os.getenv("NUM_THREADS") or os.cpu_count())
    print(
        f"assignments in pickle are for {len(assignments)} threads, we have {threads} threads"
    )
    if len(assignments) != threads or os.getenv("REDO_PARTITION"):
        sizes = tuple(meta["compressed_size"] for meta in real_meta)
        assignments = partition.massage(sizes, threads)

    for bin in assignments:
        biggest_file = max(bin, key=lambda idx: real_meta[idx]["compressed_size"])
        bin.remove(biggest_file)
        bin.insert(0, biggest_file)

    chunk_size = 1 << int(os.getenv("CHUNK_SIZE") or 20)
    first_sizes = [real_meta[bin[0]]["decompressed_size"] for bin in assignments]
    size = sum(min(math.ceil(s / chunk_size), 4) * chunk_size for s in first_sizes)
    os.environ["TOTAL_FILE_SIZE"] = str(size)
    json.dump(
        {"meta": real_meta, "assignments": assignments, "size": size},
        open("/tmp/init.json", "w"),
    )

    # tensors = _nyacomp.good_batch_decompress(fnames, shapes, dtypes, -1, -1)
    with annotate("batch_decompress"):
        tensors = _nyacomp.batch_decompress(files, assignments)
    if None in tensors:
        import pdb

        pdb.set_trace()

    tensors_iter = iter(tensors)
    with timer("torch.load"):
        with annotate("torch.load"):
            model = torch.load(path, map_location="cuda:0")
    for param, meta in zip(model.parameters(), metadata):
        if meta:
            param.data = next(tensors_iter)

    return model


def stats(times: list[int | float]) -> str:
    import statistics

    _stats = {
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times),
        "min": min(times),
    }
    return " ".join(f"{k}: {round(v, 4)}" for k, v in _stats.items())



if __name__ == "__main__":
    COMPRESS = os.getenv("COMPRESS")
    model_path = Path("./data/boneless_model.pth")
    if COMPRESS:
        with timer("from_pretrained"):
            if os.getenv("DIFFUSERS"):
                import diffusers
                model = diffusers.StableDiffusionPipeline("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16", "safety_checker"=None, local_files_only=True)
            else:
                import transformers
                model = transformers.CLIPModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
        # model = torch.load("/tmp/clip.pth", map_location="cpu")
        compress(model, model_path)
        del model
        torch.cuda.memory.empty_cache()
    # with timer("import transformers"):
    #     import transformers
    with timer("load_compressed"):
        model = load_compressed(model_path)


# memory use:
# O(1) pinned host memory and reads
# ~O(n) compressed size (max is memory ready or used by simultanious decompressions at a given point in time)
# O(n) decompressed tensors
#
# we can try to solve for low max bandwidth by packing items more closely in time, less bandwidth with higher utilisation
# have every thread finish at the same time even if it means more memory used as long as its under the memory makespan
#


# we could try using torch's memory allocator
