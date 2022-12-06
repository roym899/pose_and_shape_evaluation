import glob
import os

from torch.utils.cpp_extension import load

directory = os.path.dirname(__file__)

cpp_sources = glob.glob(f"{directory}/csrc/*.cpp")
cu_sources = glob.glob(f"{directory}/csrc/*.cu")

_C = load(
    name="dpdn_pointnet2",
    sources=cpp_sources + cu_sources,
)
