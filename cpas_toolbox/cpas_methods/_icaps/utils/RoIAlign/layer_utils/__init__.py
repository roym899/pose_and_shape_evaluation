import glob
import os

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

directory = os.path.dirname(__file__)

extensions_dir_path = os.path.join(directory, "csrc")

sources = glob.glob(os.path.join(extensions_dir_path, "*.cpp"))
sources += glob.glob(os.path.join(extensions_dir_path, "cpu", "*.cpp"))

extra_cflags = []

if torch.cuda.is_available() and CUDA_HOME is not None:
    sources += glob.glob(os.path.join(extensions_dir_path, "cuda", "*.cu"))
    extra_cflags += [
        "-DWITH_CUDA=1" "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]

_C = load(
    name="RoIAlign",
    extra_cflags=extra_cflags,
    extra_cuda_cflags=extra_cflags,
    sources=sources,
)
