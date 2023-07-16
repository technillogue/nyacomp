import ctypes
import numpy as np
import torch
import _nyacomp

def tensor_bytes(tensor: torch.Tensor) -> bytes:
    length = int(np.prod(tensor.shape).item())
    bytes_per_item = torch._utils._element_size(tensor.dtype)

    total_bytes = length * bytes_per_item

    ptr = tensor.data_ptr()
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    print("getting tensor bytes")
    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy

    return data.tobytes()


def dtype(t: torch.Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


def basic_roundtrip(tensor: torch.Tensor, fname: str = "/tmp/test.lz4") -> torch.Tensor():
    print(_nyacomp.compress(tensor_bytes(tensor), fname))
    dest = torch.empty(tensor.shape, dtype=tensor.dtype, device="cuda:0")
    print(_nyacomp.decompress(fname, dest))
    return dest




def test_basic_roundtrip():
    x = torch.rand([16])
    assert (basic_roundtrip(x) == x.cuda()).all()
    y = torch.ones([16], dtype=torch.float16)
    assert (basic_roundtrip(y) == y.cuda()).all()
    z = torch.ones([16], dtype=torch.uint8)
    assert (basic_roundtrip(z) == z.cuda()).all()


def roundtrip_new_tensor(tensor: torch.Tensor, fname: str = "/tmp/test.gz") -> torch.Tensor():
    _nyacomp.compress(tensor_bytes(tensor), fname)
    return _nyacomp.decompress_new_tensor(fname, list(tensor.shape), dtype(tensor))

def test_roundtrip_new_tensor():
    x = torch.rand([16])
    assert (roundtrip_new_tensor(x) == x.cuda()).all()
    y = torch.ones([16], dtype=torch.float16)
    assert (roundtrip_new_tensor(y) == y.cuda()).all()
    z = torch.ones([16], dtype=torch.uint8)
    assert (roundtrip_new_tensor(z) == z.cuda()).all()

def test_batch():
    x = torch.rand([512])
    fname = "/tmp/test.gz"
    _nyacomp.compress(tensor_bytes(x), fname)
    x_out = torch.empty([512], device="cuda:0")
    tensors = _nyacomp.decompress_batch_async([fname], [x_out])
    assert (x_out == x.cuda()).all()

def test_batch_new():
    x = torch.rand([512])
    fname = "/tmp/test.gz"
    _nyacomp.compress(tensor_bytes(x), fname)
    tensors = _nyacomp.decompress_batch_async_new([fname], [[512]], ["float32"])
    x_out = tensors[0]
    assert (x_out == x.cuda()).all()


#dtype = lambda t:str(t.dtype).removeprefix("torch.")
# floats = torch.rand([3])
# _nyacomp.compress(tensor_bytes(floats), "/tmp/t.gz")
# floats_out =  _nyacomp.good_batch_decompress_threadpool(["/tmp/t.gz"], [[3]], ["float32"])
# print(floats, floats_out[0])

# ints = torch.ones([3], dtype=torch.uint8)
# _nyacomp.compress(tensor_bytes(ints), "/tmp/t.gz")
# ints_out =  _nyacomp.good_batch_decompress_threadpool(["/tmp/t.gz"], [[3]], ["uint8"])
# print(ints, ints_out[0])


def good_roundtrip(tensor: torch.Tensor, fname: str = "/tmp/test.lz4") -> torch.Tensor():
    _nyacomp.compress(tensor_bytes(tensor), fname)
    results = _nyacomp.good_batch_decompress_threadpool([fname], [list(tensor.shape)], [dtype(tensor)])
    print(results[0])
    return results[0]#

    # dir = Path(path).parent / "nya"
    # state_dict = torch.load(dir / f"boneless_{Path(path).name}")

    # keys = [k for k, v in state_dict.items() if isinstance(v, dict)]
    # fnames = [f"{dir / key}.gz" for key in keys]
    # shapes = [list(state_dict[k]["shape"]) for k in keys]
    # dtypes = [str(state_dict[k]["dtype"]).split(".")[1] for k in keys]

    # tensors = _nyacomp.good_batch_decompress_threaded(fnames, shapes, dtypes)
    # return state_dict | dict(zip(keys, tensors))

def test_good_roundtrip():
    y = torch.rand([16], dtype=torch.float32)
    print("float")
    y_t = good_roundtrip(y)
    print("float:",y_t.eq(y.cuda()).all())
    assert y_t.eq(y.cuda()).all()
    x = torch.ones([16], dtype=torch.uint8)
    x_t = good_roundtrip(x)
    print("int:", x_t.eq(x.cuda()).all())

# torch.cuda.synchronize()

# test_basic_roundtrip()
# print("=============testing simple cuda-async-only batch")
# test_batch()
# print("=============testing simple cuda-async-only + cpp tensor creation batch")
# test_batch_new()
# print("=============testing roundtrip_new_tensor")
# test_roundtrip_new_tensor()
print("=============testing good_roundtrip")
test_good_roundtrip()


