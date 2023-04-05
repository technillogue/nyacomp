#!/bin/bash
set -o xtrace
set -o pipefail
set -o errexit
# git clone ...
# curl -O https://developer.download.nvidia.com/compute/nvcomp/2.6.1/local_installers/nvcomp_2.6.1_x86_64_12.x.tgz
# # replace setup.py paths
# python3.10 -m pip install torch pybind11
python3.10 setup.py bdist_wheel
# the paths need to be dynamic
# basically all of ~/.local/lib/python3.10/site-packages/torch/lib/*
LD_LIBRARY_PATH="/opt/_internal/cpython-3.10.9/lib/python3.10/site-packages/torch/lib:/workdir/nvcomp/lib:$LD_LIBRARY_PATH" auditwheel -vvv repair \
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
dist/nyacomp-0.0.1-cp310-cp310-linux_x86_64.whl

# better name:
# nyacomp-0.0.1-cp310-manylinux2014_x86_64.whl
mkdir -p wheelhouse/check
cd wheelhouse/check || exit
unzip -o ../nyacomp-0.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl _nyacomp.cpython-310-x86_64-linux-gnu.so
patchelf --add-rpath '$ORIGIN/torch/lib' _nyacomp.cpython-310-x86_64-linux-gnu.so
# this has to be kept in sync with the torch version
patchelf --replace-needed libcudart.so.11.0 libcudart-a7b20f20.so.11.0  _nyacomp.cpython-310-x86_64-linux-gnu.so
# if building for release, strip debug symbols from each binary in nyacomp.libs
if [[ "$1" == "release"]]; then
    unzip -o ../nyacomp-0.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl nyacomp.libs
    find . -name '*.so' | xargs strip --strip-debug
    zip -o ../nyacomp-0.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl nyacomp.libs
fi
zip -o ../nyacomp-0.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl _nyacomp.cpython-310-x86_64-linux-gnu.so
