# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools import setup

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)


NVCOMP_INCLUDE_DIR = "/home/sylv/dryad/fast/py-nvcomp/_nvcomp/include"
NVCOMP_LIB_NAME = "nvcomp"
NVCOMP_LIB_DIR = "/home/sylv/dryad/fast/py-nvcomp/_nvcomp/lib"

# CUDA_INCLUDE_DIR = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include'
CUDA_INCLUDE_DIR = "/usr/local/cuda-12.1/include"
CUDA_LIB_NAME = "cudart"
# CUDA_LIB_DIR = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/lib/x64'
CUDA_LIB_DIR = "/usr/local/cuda-12.1/lib64"

ext_modules = [
    CppExtension(
        "_nyacomp",
        ["src/main.cpp"],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
        include_dirs=["/home/sylv/dryad/fast/py-nvcomp/_nvcomp/include", "/usr/local/cuda-12.1/include"],
        libraries=["nvcomp", "cudart"],
        library_dirs=["/home/sylv/dryad/fast/py-nvcomp/_nvcomp/lib", "/usr/local/cuda-12.1/lib64"],
    ),
]

setup(
    name="nyacomp",
    version=__version__,
    author="sylv",
    author_email="",
    #url="https://github.com/pybind/python_example",
    description="python bindings for using nvcomp with torch",
    #long_description="",
    ext_modules=ext_modules,
    #extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.7",
    packages=["nyacomp"],
    package_data={"nyacomp": ["lib/libnvcomp.so"]},
)
