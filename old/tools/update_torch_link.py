from pathlib import Path
import torch
from subprocess import run

rt = next((Path(torch.__file__).parent / "lib").glob("libcudart*"))
nyacomp = "_nyacomp.cpython-310-x86_64-linux-gnu.so"
run(
    f"patchelf --replace-needed libcudart-a7b20f20.so.11.0 {rt} {nyacomp}",
    shell=True,
)
