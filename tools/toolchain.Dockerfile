FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
RUN --mount=type=cache,target=/var/cache/apt apt-get update && apt-get install -yy libnvidia-compute-515-server python3.11 patchelf zip unzip curl build-essential gcc libpython3.11-dev git
RUN curl -L https://bootstrap.pypa.io/get-pip.py|python3.11
#COPY ./torch-2.1.0a0+git3af011b-cp310-cp310-linux_x86_64.whl ./torch-2.1.0a0+git3af011b-cp310-cp310-linux_x86_64.whl
#COPY ./torch-2.0.0a0+gite9ebda2-cp311-cp311-linux_x86_64.whl ./torch-2.0.0a0+gite9ebda2-cp311-cp311-linux_x86_64.whl
#RUN pip install ./torch-2.0.0a0+gite9ebda2-cp311-cp311-linux_x86_64.whl
RUN --mount=type=cache,target=/root/.cache/pip pip install \
  pybind11 --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.0.1+cu118
# this is a direct wheel but, this fetches something mangled cudart, instead of uncommented depending on nvidia-cuda-runtime(?)
#'https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp311-cp311-linux_x86_64.whl#sha256=143b6c658c17d43376e2dfbaa2c106d35639d615e5e8dec4429cf1e510dd8d61'
WORKDIR /workdir
