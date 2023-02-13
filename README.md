# Categorical Pose and Shape Evaluation Toolbox
[![PyPI Release](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/publish_release.yaml/badge.svg)](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/publish_release.yaml) [![PyTest](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/pytest.yaml/badge.svg)](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/pytest.yaml) [![Docs](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/build_docs.yaml/badge.svg)](https://github.com/roym899/pose_and_shape_evaluation/actions/workflows/build_docs.yaml)

CPAS Toolbox is a package for evaluation of categorical pose and shape estimation methods. It contains metrics, datasets and methods.
Visit the [documentation](https://roym899.github.io/pose_and_shape_evaluation/) for detailed usage instructions and API reference.


## Installation
```bash
pip install cpas_toolbox
```

## Citation
If you find this library useful in your research, consider citing [our conference paper](https://link.springer.com/chapter/10.1007/978-3-031-22216-0_25) or the [extended journal version (preprint)](https://arxiv.org/abs/2301.08147):
```
@inproceedings{bruns2023evaluation,
  title={On the evaluation of RGB-D-based categorical pose and shape estimation},
  author={Bruns, Leonard and Jensfelt, Patric},
  booktitle={Intelligent Autonomous Systems 17: Proceedings of the 17th International Conference IAS-17},
  pages={360--377},
  year={2023},
  organization={Springer}
}
@article{bruns2023rgb,
  title={RGB-D-Based Categorical Object Pose and Shape Estimation: Methods, Datasets, and Evaluation},
  author={Bruns, Leonard and Jensfelt, Patric},
  journal={arXiv preprint arXiv:2301.08147},
  year={2023}
}
```

## Development
- Use `pip install -e .` to install the package in editable mode
- Use `pip install -r requirements-dev.txt` to install dev tools
- Use `pytest -rf --cov=cpas_toolbox --cov-report term-missing tests/` to run tests and check code coverage
