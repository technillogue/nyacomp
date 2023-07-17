#!/bin/bash
set -o xtrace
set -o pipefail
set -o errexit
# git clone ...
# curl -O https://developer.download.nvidia.com/compute/nvcomp/2.6.1/local_installers/nvcomp_2.6.1_x86_64_12.x.tgz
# # replace setup.py paths
# python3.11 -m pip install torch pybind11
# compile it targeting manylinux
python3.11 setup.py bdist_wheel
# bundle us and nvcomp, but exclude everything torch uses
# the paths need to be dynamic
# basically all of ~/.local/lib/python3.11/site-packages/torch/lib/*
LD_LIBRARY_PATH="/opt/_internal/cpython-3.11.9/lib/python3.11/site-packages/torch/lib:/workdir/nvcomp/lib:$LD_LIBRARY_PATH" auditwheel -vvv repair \
--exclude libc10_cuda.so \
--exclude libc10.so \
--exclude libcublasLt.so.11 \
--exclude libcublasLt.so.12 \
--exclude libcublas.so.11 \
--exclude libcublas.so.12 \
--exclude libcudart-5a077122.so.12.0.107 \
--exclude libcudart-a7b20f20.so.11.0 \
--exclude libcudart-e409450e.so.11.0 \
--exclude libcudart.so.11.0 \
--exclude libcudart.so.11.7.60 \
--exclude libcudart.so.12 \
--exclude libcudart.so.12.0 \
--exclude libcudart.so.12.0.107 \
--exclude libcudart.so.12.7.60 \
--exclude libcudnn.so.8 \
--exclude libgomp-a34b3233.so.1 \
--exclude libgomp.so.1 \
--exclude libnvToolsExt-24de1d56.so.1 \
--exclude libshm.so \
--exclude libtorch_cpu.so \
--exclude libtorch_cuda_cpp.so \
--exclude libtorch_cuda_cu.so \
--exclude libtorch_cuda.so \
--exclude libtorch_python.so \
--exclude libtorch.so \
--exclude libnvidia-ml.so.515.43.04 \
dist/nyacomp-0.0.1-cp311-cp311-linux_x86_64.whl

# better name:
# nyacomp-0.0.1-cp311-manylinux2014_x86_64.whl
mkdir -p wheelhouse/check
cd wheelhouse/check || exit
unzip -o ../nyacomp-0.0.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl _nyacomp.cpython-311-x86_64-linux-gnu.so
patchelf --add-rpath '$ORIGIN/torch/lib' _nyacomp.cpython-311-x86_64-linux-gnu.so
# this has to be kept in sync with the torch version
#RT_LIB=$(python3.11 -c 'import pathlib as p, torch; print(next((p.Path(torch.__file__).parent / "lib").glob("libcudart*")).name)')
RT_LIB="libcudart-d0da41ae.so.11.0"
patchelf --replace-needed libcudart.so.11.0 $RT_LIB _nyacomp.cpython-311-x86_64-linux-gnu.so
# if building for release, strip debug symbols from each binary in nyacomp.libs
if [[ "$1" == "release" ]]; then
    strip --strip-debug _nyacomp.cpython-311-x86_64-linux-gnu.so
    unzip -o ../nyacomp-0.0.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 'nyacomp.libs/*'
    find . -name '*.so' | xargs strip --strip-debug
    zip -o ../nyacomp-0.0.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl nyacomp.libs/*
fi
zip -o ../nyacomp-0.0.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl _nyacomp.cpython-311-x86_64-linux-gnu.so
