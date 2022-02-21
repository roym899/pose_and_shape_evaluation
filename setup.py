import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cpas_toolbox",
    version="0.1.0",
    author="Leonard Bruns",
    author_email="roym899@gmail.com",
    description="Toolbox to evaluate categorical pose and shape estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/roym899/pose_and_shape_evaluation",
    packages=setuptools.find_packages(),
    package_data={"": ["config/*"]},
    install_requires=[
        "gdown",
        "matplotlib",
        "numpy",
        "open3d",
        "opencv-python-headless",
        "Pillow",
        "requests",
        "scipy",
        "tikzplotlib",
        "torch",
        "torchvision",
        "tqdm",
        "trimesh",
        "yoco",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.6",
)
