# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools import setup

__version__ = "0.0.3"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)


# class CustomBuildExtension(BuildExtension):
#     def build_extension(self, ext):
#         # Change the extension type to an executable
#         ext.target_ext = ""

#         # Call the superclass method to build the extension
#         super().build_extension(ext)


NVCOMP_LIB_NAME = "nvcomp"
NVCOMP_INCLUDE_DIR = "nvcomp/include"
NVCOMP_LIB_DIR = "nvcomp/lib"

CUDA_LIB_NAME = "cudart"
CUDA_INCLUDE_DIR = "/usr/local/cuda/include"
CUDA_LIB_DIR = "/usr/local/cuda/lib64"

ext_modules = [
    CppExtension(
        "_nyacomp",
        ["src/main.cpp"],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
        include_dirs=[NVCOMP_INCLUDE_DIR, CUDA_INCLUDE_DIR],
        libraries=[CUDA_LIB_NAME, NVCOMP_LIB_NAME],
        library_dirs=[NVCOMP_LIB_DIR, CUDA_LIB_DIR],
        extra_compile_args=["-fvisibility=hidden", "-std=c++14"],
    ),
]

setup(
    name="nyacomp",
    version=__version__,
    author="sylv",
    description="python bindings for using nvcomp with torch",
    ext_modules=ext_modules,
    # extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.7",
    packages=["nyacomp"],
    # package_dir={"": "."},
    package_data={
        "nyacomp": [
            "lib/libnvcomp_bitcomp.so",
            "lib/libnvcomp_gdeflate.so",
            "lib/libnvcomp.so",
        ]
    },
    include_package_data=True,
)
# actually depends on torch (specific version of torch). optional: nvtx, humanize. also requires transformers or diffusers as appropriate
