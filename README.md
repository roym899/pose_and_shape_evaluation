# Categorical Pose and Shape Evaluation Toolbox
[![PyPI Release](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/publish_release.yaml/badge.svg)](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/publish_release.yaml) [![PyTest](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/pytest.yaml/badge.svg)](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/pytest.yaml) [![Docs](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/build_docs.yaml/badge.svg)](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/build_docs.yaml)

CPAS Toolbox is a package for evaluation of categorical pose and shape estimation methods. It contains metrics, and wrappers for datasets and methods.
Visit the [documentation](https://roym899.github.io/pose_and_shape_evaluation/) for detailed usage instructions and API reference.


## Installation
```bash
pip install cpas_toolbox
```

## Development
- Use `pip install -e .` to install the package in editable mode
- Use `pip install -r requirements-dev.txt` to install dev tools
- Use `pytest -rf --cov=cpas_toolbox --cov-report term-missing tests/` to run tests and check code coverage
