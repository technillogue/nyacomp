# pylint: disable=import-outside-toplevel,wrong-import-position,c-extension-no-member
import importlib.abc
import importlib.util
import os
import sys

if os.uname().nodename == "decid":
    # oomkill us before killing chrome tabs 
    open("/proc/self/oom_score_adj", "w").write("1000")

os.environ["HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if not os.getenv("NO_HIDE_MODULES"):
    hidden = {"jax", "flax", "accelerate", "wandb"}

    def find_spec(name: str, package: "Any" = None) -> "Any":
        if name in hidden:
            return None
        return orig_find_spec(name, package)

    class CustomFinder(importlib.abc.MetaPathFinder):
        # pylint: disable=unused-argument
        def find_spec(self, fullname: str, path: str, target: "Any" = None) -> "Any":
            if fullname in hidden:
                raise ImportError(f"{fullname} is blocked and cannot be imported.")

    orig_find_spec = importlib.util.find_spec
    importlib.util.find_spec = find_spec
    sys.meta_path.insert(0, CustomFinder())

import contextlib
import time


@contextlib.contextmanager
def timer(msg: str) -> "Iterator[None]":
    start = time.time()
    yield
    print(f"{msg} took {time.time() - start:.3f}s")


if os.getenv("SYMLINK_CUDART"):
    # torch bundles cudart with a mangled name
    # nyacomp is compiled against a specific mangled name
    # the rpath expects _nyacomp.so to be at the top level of site-packages and torch/lib/* to be where libcuda and libtorch are
    # we expect we copied site-packages into the current directory
    # a symlink is enough to support versions of torch with a different libcudart
    os.system("ln -s torch/lib/libcudart* torch/lib/libcudart-d0da41ae.so.11.0")

with timer("import _nyacomp"):
    import _nyacomp

if not os.getenv("NO_PRELOAD"):
    path = os.getenv("PRELOAD_PATH", "data/nya/meta.csv")
    with timer("proceeding with importing; launching decompression"):
        decompressor = _nyacomp.AsyncDecompressor(path)
else:
    decompressor = None


with timer("stdlib imports"):
    import ctypes
    import io
    import itertools as it
    import gc
    import json
    import math
    import pickle
    import pickletools
    import threading
    import timeit
    from pathlib import Path
    from typing import Any, Iterator, Union, Sequence

    try:
        from nyacomp import partition
    except ImportError:
        import partition

NUM_THREADS = int(os.getenv("NUM_THREADS") or len(os.sched_getaffinity(0)))

# FIXME: make the entire annotate thing configurable
# and integrate it with timer
# just have loglevel

try:
    with timer("nvtx"):
        from nvtx import annotate
except ImportError:

    class annotate:
        def __init__(self, _: str) -> None:
            pass

        def __enter__(self, *args: "Any", **kwargs: "Any") -> "Any":
            pass

        def __exit__(self, *args: "Any", **kwargs: "Any") -> "Any":
            pass

        def __call__(self, func: "Any") -> "Any":
            return func


try:
    from humanize.filesize import naturalsize as natsize
except ImportError:
    natsize = str  # type: ignore


def tensor_size(tensor: "torch.Tensor") -> int:
    return tensor.nelement() * tensor.element_size()


def tensor_bytes(tensor: "torch.Tensor") -> bytes:
    total_bytes = tensor_size(tensor)

    ptr = tensor.data_ptr()
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy
    return data.tobytes()


Tensory = Union["torch.Tensor", "torch.nn.Parameter"]
if os.getenv("DOWNLOAD"):
    HOST = os.getenv("HOST", "http://localhost:8080")
else:
    HOST = ""


def compress_parameter(param: Tensory, path: Path) -> tuple[dict, int, int]:
    data = tensor_bytes(param.data.detach().cpu())
    size = len(data)

    # Helper function to save raw data and adjust meta
    def save_raw(p: Path, d: bytes, m: dict) -> tuple[dict, int, int]:
        p.with_suffix(".raw").open("wb").write(d)
        m |= {"filename": f'{HOST}/{p.with_suffix(".raw")}', "compressed_size": size}
        return m, size, size

    if size < (1 << 1):
        return {}, size, size
    if os.getenv("UNCOMPRESSED"):
        return save_raw(path, data, {})

    if path.exists() and not os.getenv("RECOMPRESS"):
        print(f"skipping already existing parameter at {path}")
        new_size = path.stat().st_size
    else:
        print(f"compressing parameter to {path}")
        new_size = _nyacomp.compress(data, str(path))
    meta = {
        "filename": f'{HOST}/{path}',
        "shape": list(param.shape),
        "dtype": str(param.dtype).removeprefix("torch."),
        "decompressed_size": size,
        "compressed_size": new_size,
    }
    # torch.nn.parameter.UninitializedParameter
    param.data = torch.tensor([], dtype=param.dtype)
    if new_size > size:
        print("bad compression ratio, replacing with uncompressed", path)
        path.unlink()
        return save_raw(path, data, meta)
    return meta, size, new_size


default_path = Path("./boneless_model.pth")


def ints(i: list[int]) -> str:
    # in the case of shape [], we need something to not fuck up our csv, C++ unpacking is odd
    # or nvm
    return ";".join(map(str, i))  # or ";"


def to_csv(meta: list[dict], bins: list[list[int]], f: str) -> None:
    ass = ",".join(map(ints, bins))
    info = [
        [
            m["filename"],
            ints(m["shape"]),
            m["dtype"],
            str(m["decompressed_size"]),
            str(m["compressed_size"]),
        ]
        for m in meta
        if m
    ]
    lines = list(map(" ".join, info))
    open(f, "w").write("\n".join([ass] + lines))


Compressable = Union["torch.nn.Module", dict]
IdxTensor = tuple[int, "torch.Tensor"]

MERGE_INFO_FNAME = "merged_tensors.csv"
# f"{group[0][1].shape}:" +


def get_bins(group: list[IdxTensor], n_splits: int) -> list[list[IdxTensor]]:
    bin_sizes = [0 for _ in range(n_splits)]
    # preserve the order of bins, but make sure we would get [3, 2, 2] instead of [3, 3, 1]
    for i in range(len(group)):
        bin_sizes[i % n_splits] += 1
    # [2, 3, 3] -> [2, 5, 8]
    bin_offsets = list(it.accumulate(bin_sizes))
    bins = [
        sorted(group[start:end])
        # -> [(0, 2), (2, 5), (5, 8)]
        for start, end in zip([0] + bin_offsets, bin_offsets)
    ]
    return bins


def merge_tensors(tensors: Sequence[Tensory]) -> tuple[list["torch.Tensor"], str]:
    real_parameter_idx = enumerate(tensors)  # [(idx, tensor), ...]
    grouper = lambda x: x[1].shape
    maxsize = max(map(tensor_size, tensors))
    subgroups = []

    for shape, group in it.groupby(sorted(real_parameter_idx, key=grouper), grouper):
        group = list(group)
        groupsize = tensor_size(group[0][1]) * len(group)
        if groupsize >= maxsize:
            # split the group into n groups such that the splits are close to equal size
            # and each split is less than maxsize
            n_splits = math.ceil(groupsize / maxsize)
            bins = get_bins(group, n_splits)
            subgroups.extend(sorted(bins))
        else:
            subgroups.append(sorted(group))
    # in case of 0-dim tensors (scalars), we add a dimension before concatenating
    # then remove it after splitting
    merged_tensors = [
        torch.cat([torch.unsqueeze(param[1], 0) for param in group], 0)
        for group in subgroups
    ]
    info = "\n".join(",".join(str(param[0]) for param in group) for group in subgroups)
    return merged_tensors, info


def split_tensors(tensors: list["torch.Tensor"], info: str) -> list["torch.Tensor"]:
    merges = [list(map(int, line.split(","))) for line in info.split("\n")]
    unmerged = [
        pair
        for tensor, idxs in zip(tensors, merges)
        for pair in zip(idxs, torch.tensor_split(tensor, len(idxs), dim=0))
    ]
    # sort by index to recover the original order
    return [
        torch.squeeze(tensor, 0) for _, tensor in sorted(unmerged, key=lambda x: x[0])
    ]


def compress(model: Compressable, path: str | Path = default_path) -> float:
    global torch
    import numpy as np
    import torch

    sys.modules[__name__].np = np  # import here so tensor_bytes can find it

    if isinstance(path, str):
        path = Path(path)
    if isinstance(model, torch.nn.Module):
        # parameters = list(model.named_parameters()))
        orig_parameters = list(model.parameters())
    elif isinstance(model, dict):
        # state dict
        orig_parameters = list(sorted(model.values()))
    else:
        # pipeline
        orig_parameters = get_pipeline_params(model)

    if isinstance(path, str):
        path = Path(path)
    dir = path.parent / "nya"
    dir.mkdir(exist_ok=True)

    parameters, info = merge_tensors(orig_parameters)  # hmm
    open(path.parent / MERGE_INFO_FNAME, "w").write(info)

    total_size = 0.0
    total_compressed_size = 0
    meta = []

    for i, param in enumerate(parameters):
        param_path = dir / f"{i}.gz"
        param_meta, size, new_size = compress_parameter(param, param_path)
        meta.append(param_meta)
        total_size += size
        total_compressed_size += new_size

    for param in orig_parameters:
        param.data = torch.tensor([], dtype=param.dtype)

    sizes = tuple(param_meta["compressed_size"] for param_meta in meta if param_meta)
    assignments = partition.massage(sizes, NUM_THREADS)

    to_csv(meta, assignments, str(dir / "meta.csv"))
    meta.append(assignments)  # type: ignore

    pickle.dump(meta, open(str(dir / "metadata.pkl"), "wb"))
    print("overall compression ratio:", total_compressed_size / total_size)
    print("saving boneless model to ", path)

    torch.save(model, str(path))
    return total_compressed_size / total_size

def empty_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()

def compress_pickle(model: Compressable, path: str | Path = default_path) -> float:
    global torch
    import numpy as np
    import torch

    sys.modules[__name__].np = np  # import here so tensor_bytes can find it

    if isinstance(path, str):
        path = Path(path)
    dir = path.parent / "nya"
    dir.mkdir(exist_ok=True)

    orig_tensors: list[torch.Tensor] = []

    def persistent_id(obj: Any) -> int | None:
        # don't compress cpu tensors
        if isinstance(obj, torch.Tensor) and obj.device.type != "cpu":
            i = len(orig_tensors)
            orig_tensors.append(obj.to("cpu"))
            return i

        return None

    buf = io.BytesIO()
    pickler = pickle.Pickler(buf, protocol=5)
    pickler.persistent_id = persistent_id
    pickler.dump(model)
    del model
    empty_cache()
    buf.seek(0)
    # removes unused PUTs
    open(path, "wb").write(pickletools.optimize(buf.read()))
    parameters, info = merge_tensors(orig_tensors)  # hmm
    del orig_tensors
    empty_cache()
    open(path.parent / MERGE_INFO_FNAME, "w").write(info)

    total_size = 0.0
    total_compressed_size = 0
    meta = []

    for i, param in enumerate(parameters):
        param_path = dir / f"{i}.gz"
        param_meta, size, new_size = compress_parameter(param, param_path)
        meta.append(param_meta)
        total_size += size
        total_compressed_size += new_size
        del param

    # at this point we can concatenate files based on thread assignment for upload

    sizes = tuple(param_meta["compressed_size"] for param_meta in meta if param_meta)
    assignments = partition.massage(sizes, NUM_THREADS)

    to_csv(meta, assignments, str(dir / "meta.csv"))
    meta.append(assignments)  # type: ignore

    pickle.dump(meta, open(str(dir / "metadata.pkl"), "wb"))  # unnecessary?
    print("overall compression ratio:", total_compressed_size / total_size)
    print("saved boneless model to ", path)
    return total_compressed_size / total_size


def get_pipeline_params(
    pipeline: "diffusers.DiffusionPipeline",
) -> list["torch.nn.Parameter"]:
    # https://github.com/huggingface/diffusers/blob/v0.7.0/src/diffusers/pipeline_utils.py#L174-L180
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_utils.py#L493-L497
    exclude = {"_class_name", "_diffusers_version", "_module"}
    sub_models = [
        getattr(pipeline, pipeline_component_name)
        for pipeline_component_name in pipeline.config.keys()
        if pipeline_component_name not in exclude
    ]
    return [
        param
        for sub_model in sub_models
        if sub_model and isinstance(sub_model, torch.nn.Module)
        for param in sub_model.parameters()
    ]


def get_args(path: Path) -> tuple[list[_nyacomp.CompressedFile], list[list[int]]]:
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
            meta["compressed_size"],
        )
        for meta in real_meta
    ]

    print(
        f"assignments in pickle are for {len(assignments)} threads, we have {NUM_THREADS} threads"
    )
    if len(assignments) != NUM_THREADS or os.getenv("REDO_PARTITION"):
        sizes = tuple(meta["compressed_size"] for meta in real_meta)
        assignments = partition.massage(sizes, NUM_THREADS)

    chunk_size = 1 << int(os.getenv("CHUNK_SIZE") or 20)
    first_sizes = [real_meta[bin[0]]["decompressed_size"] for bin in assignments]
    size = sum(min(math.ceil(s / chunk_size), 4) * chunk_size for s in first_sizes)
    os.environ["TOTAL_FILE_SIZE"] = str(size)
    json.dump(
        {"meta": real_meta, "assignments": assignments, "size": size},
        open("/tmp/init.json", "w"),
    )
    return files, assignments


def get_tensors(path: Path) -> list["torch.Tensor"]:
    if decompressor:
        try:
            with timer("decompressor.get"):
                tensors = decompressor.get()
            return tensors
        except (RuntimeError, ValueError) as e:
            print(f"decompressor.get errored: {e}")
            raise e
            sys.exit(1)
    with timer("batch_decompress"):
        with annotate("batch_decompress"):
            files, assignments = get_args(path)
            tensors = _nyacomp.batch_decompress(files, assignments)
    return tensors


@annotate("load_compressed")
def load_compressed(path: str | Path = default_path) -> Compressable:
    if isinstance(path, str):
        path = Path(path)
    print("started load_compressed")
    # with timer("import huggingface lib"):
    #     if os.getenv("ENV") == "PROD":
    #         import diffusers
    #     else:
    #         import transformers
    with timer("import torch"):
        global torch
        import torch
    with timer("boneless torch.load"):
        with annotate("torch.load"):
            print("boneless torch.load")
            model = torch.load(path)

    with timer("load tensors"):
        tensors = get_tensors(path)
    if None in tensors:
        import pdb

        pdb.set_trace()

    real_tensors = split_tensors(tensors, open(path.parent / MERGE_INFO_FNAME).read())

    tensors_iter = iter(real_tensors)
    empty_size = torch.Size([0])
    with timer("setting params"):
        if isinstance(model, torch.nn.Module):
            params = list(model.parameters())
        else:
            params = get_pipeline_params(model)
        for param in params:
            if param.data.size() == empty_size:
                param.data = next(tensors_iter)

    assert next(tensors_iter, None) is None, "used tensors remaining"

    return model


@annotate("load_compressed_pickle")
def load_compressed_pickle(path: str | Path = default_path) -> Compressable:
    if isinstance(path, str):
        path = Path(path)
    print("started load_compressed")
    with timer("import torch"):
        global torch
        import torch

    lazy_tensors = {}

    def persistent_load(id: int) -> "torch.Tensor":
        # this could be a device=meta, but we want to set t.data later on
        lazy_tensors[id] = torch.tensor([])
        return lazy_tensors[id]

    unpickler = pickle.Unpickler(open(path, "rb"))
    unpickler.persistent_load = persistent_load
    with timer("unpickling"):
        model = unpickler.load()
    with timer("load tensors"):
        tensors = get_tensors(path)

    real_tensors = split_tensors(tensors, open(path.parent / MERGE_INFO_FNAME).read())
    with timer("setting params"):
        for idx, tensor in enumerate(real_tensors):
            lazy_tensors[idx].data = tensor
    return model


@contextlib.contextmanager
def cleanup() -> None:
    prev_size = torch.cuda.memory.memory_reserved()
    yield
    gc.collect()
    used_size = torch.cuda.memory.memory_reserved()
    torch.cuda.memory.empty_cache()
    freed_size = torch.cuda.memory.memory_reserved()
    print(
        "previous torch mem",
        natsize(prev_size),
        "used mem",
        natsize(used_size),
        "mem after clearing cache",
        natsize(freed_size),
    )
    # if torch.cuda.memory.memory_reserved() > prev_size:
    #     import pdb
    #     pdb.set_trace()


def with_cleanup(path: Path) -> None:
    with cleanup():
        model = load_compressed_pickle(path)
        # model.scheduler.alphas_cumprod.to("cpu")
        # model.text_encoder.text_model.embeddings.to("cpu")
        # model("horse ").images[0].save("/tmp/out.png")


def stats(times: list[int | float]) -> str:
    import statistics

    _stats = {
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times),
        "min": min(times),
    }
    return " ".join(f"{k}: {round(v, 4)}" for k, v in _stats.items())


if __name__ == "__main__" or os.getenv("RUN_MAIN"):
    with timer("import torch"):
        import torch
    COMPRESS = os.getenv("COMPRESS")
    if os.getenv("ENV") == "PROD" or os.getenv("DIFFUSERS"):
        model_path = Path("./data/boneless_model.pth")
    else:
        model_path = Path("./data/boneless_clip.pth")
    if COMPRESS:
        with timer("from_pretrained"):
            if os.getenv("DIFFUSERS") or os.getenv("ENV") == "PROD":
                model = torch.load("/tmp/sdxl_bundle_raw.pth")
                # import diffusers

                # thing here about AIT
                # needs to be compressed for the same diffusers version
                # (or use state dict...)
                # model = diffusers.DiffusionPipeline.from_pretrained(
                #     # "CompVis/stable-diffusion-v1-4",
                #     "/home/sylv/r8/cog-sdxl/sdxl-cache",
                #     torch_dtype=torch.float16,
                #     variant="fp16",
                #     use_safetensors=True,
                #     # safety_checker=None,
                #     local_files_only=True,
                # )
                # # unfortunately, this is necessary to learn that the scheduler remains on the cpu
                # model.to("cuda")
            else:
                import transformers

                model = transformers.CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch16",  # local_files_only=True
                )
        torch.save(model, "/tmp/model.pth")
        # model = torch.load("/tmp/clip.pth", map_location="cpu")
        compress_pickle(model, model_path)
        del model
        torch.cuda.memory.empty_cache()
    # with timer("import transformers"):
    #     import transformers
    if os.getenv("PROF"):
        if os.getenv("GOOD"):
            with cleanup():
                with timer("load_compressed"):
                    load_compressed_pickle(model_path)
        if os.getenv("TORCH"):
            with timer("torch.load"):
                # import diffusers
                # model = diffusers.StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16", safety_checker=None, local_files_only=True)
                torch.load("/tmp/sdxl_bundle_raw.pth", map_location="cuda:0")
        sys.exit(0)
    os.environ["NAME"] = name = "run-" + str(int(time.time()))
    times = [
        timeit.timeit("with_cleanup(model_path)", number=1, globals=globals())
        for i in range(5)
    ]
    runs = [
        run
        for run in map(json.loads, open("/tmp/stats.json"))
        if run.get("name") == name
    ]
    print(os.environ["NAME"], " load_compressed: ", stats(times))
    processing = [run["elapsed_time"] for run in runs]
    print("inner processing: ", stats(processing))
    copy_time = [run["total_copy_time"] for run in runs]
    print("copy: ", stats(copy_time))
    decomp_time = [run["total_decomp_time"] for run in runs]
    print("decomp: ", stats(decomp_time))
    read_time = [run["total_read_time"] for run in runs]
    print("read: ", stats(read_time))


# memory use:
# O(1) pinned host memory and reads
# ~O(n) compressed size (max is memory ready or used by simultanious decompressions at a given point in time)
# O(n) decompressed tensors
#
# we can try to solve for low max bandwidth by packing items more closely in time, less bandwidth with higher utilisation
# have every thread finish at the same time even if it means more memory used as long as its under the memory makespan
#


# we could try using torch's memory allocator
#
#
# it would be really nice to hook into persistent_load, replacing the unpickler in torch.load
# unless smth diff with aitemplate
