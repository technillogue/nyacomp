def compress(data: bytes, filename: str, chunk_size_exponent: int) -> int:
    ...

def decompress(filename: str, shape: list[int], dtype: str) -> "torch.Tensor":
    ...


class CompressedFile:
    def __init__(self, filename: str, shape: list[int], dtype: str, decompressed_size: int) -> None:
        ...

def batch_decompress(files: list[CompressedFile], assignments: list[list[int]]) -> list["torch.Tensor"]:
    ...

class AsyncDecompressor:
    def __init__(self, fname: str) -> None:
        ...

    def get(self) -> list["torch.Tensor"]:
        ...
