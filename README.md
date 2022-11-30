# Categorical Pose and Shape Evaluation Toolbox
[![PyPI Release](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/publish_release.yaml/badge.svg)](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/publish_release.yaml) [![PyTest](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/pytest.yaml/badge.svg)](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/pytest.yaml) [![Docs](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/build_docs.yaml/badge.svg)](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/build_docs.yaml)

CPAS Toolbox is a package for evaluation of categorical pose and shape estimation methods. It contains metrics, datasets and methods.
Visit the [documentation](https://roym899.github.io/pose_and_shape_evaluation/) for detailed usage instructions and API reference.


## Installation
```bash
pip install cpas_toolbox
```

## Citation
If you find this library useful in your research, consider citing [our publication](https://arxiv.org/abs/2202.10346):
```
@article{bruns2022evaluation,
  title={On the Evaluation of {RGB-D}-based Categorical Pose and Shape Estimation},
  author={Bruns, Leonard and Jensfelt, Patric},
  journal={arXiv preprint arXiv:2202.10346},
  year={2022}
}
```

## Development
- Use `pip install -e .` to install the package in editable mode
- Use `pip install -r requirements-dev.txt` to install dev tools
- Use `pytest -rf --cov=cpas_toolbox --cov-report term-missing tests/` to run tests and check code coverage
