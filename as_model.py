import ctypes
import os
import pickle
import numpy as np
import torch
import python_example as nvcomp

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
    bytes_per_item = _SIZE[tensor.dtype]

    total_bytes = length * bytes_per_item

    ptr = tensor.data_ptr()
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))
    print("getting tensor bytes")
    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy

    return data.tobytes()




def compress(model: torch.nn.Module, name: str = "model.pth") -> None:
    parameters = list(model.parameters())

    for i, param in enumerate(parameters):
        data = tensor_bytes(param.data.detach())
        print(f"compressing parameter {i}")
        nvcomp.compress(data, f"tensors/{i}.lz4")

    meta = [{"shape": param.shape, "dtype": param.dtype} for param in parameters]
    pickle.dump(meta, open("tensors/metadata.pkl", "wb"))

    for param in parameters:
        param.data = torch.tensor([], dtype=param.dtype)

    torch.save(model, name)
    print(os.stat(name).st_size)


def load_compressed(fname: str = "model.pth") -> torch.nn.Module:
    model = torch.load(fname)
    metadata = pickle.load(open("tensors/metadata.pkl", "rb"))
    params = list(model.parameters())
    for i, param in enumerate(params):
        param.data = torch.empty(
            metadata[i]["shape"], dtype=metadata[i]["dtype"], device="cuda:0"
        )
        nvcomp.decompress(f"tensors/{i}.lz4", param.data, str(metadata[i]["dtype"]).split(".")[1])
    return model

model = torch.load("/home/sylv/dryad/sprkpnt/vqgan/predict/reaction_predictor_no_gauss.pth", map_location="cuda").eval()
print("loaded model")
#compress(model)
new_model = load_compressed().eval()

test_input = torch.zeros([768], device="cuda:0")
print("zeros:", new_model(test_input).eq(model(test_input)))
test_input = torch.ones([768], device="cuda:0")
print("ones:", new_model(test_input).eq(model(test_input)))

