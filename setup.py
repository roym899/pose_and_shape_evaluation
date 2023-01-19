import sys

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "easydict",  # required by iCaps
    "filterpy",  # required by iCaps
    "gdown",
    "matplotlib<3.6",  # see https://github.com/texworld/tikzplotlib/pull/558
    "ninja",
    "numba",  # required by iCaps
    "numpy",
    "open3d",
    "opencv-python-headless",
    "Pillow",
    "plyfile",  # required by iCaps
    "requests",
    "scipy",
    "scikit-image",
    "sdfest",
    "tikzplotlib",
    "tqdm",
    "torch",
    "torchvision",
    "transforms3d",  # required by iCaps
    "trimesh",
    "yoco",
]

if sys.version_info[0] <= 3 and sys.version_info[1] < 8:
    install_requires.append("typing_extensions")

setuptools.setup(
    name="cpas_toolbox",
    version="1.0.0",
    author="Leonard Bruns",
    author_email="roym899@gmail.com",
    description="Toolbox to evaluate categorical pose and shape estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roym899/pose_and_shape_evaluation",
    packages=setuptools.find_packages(),
    package_data={
        "": [
            "**/*.cpp",
            "**/*.cu",
            "**/*.json",
            "**/*.h",
            "**/*.yaml",
            "**/*.yml",
        ]
    },
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.7",
)
