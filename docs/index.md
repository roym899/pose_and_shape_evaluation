# Getting started

CPAS Toolbox is a package for evaluation of categorical pose and shape estimation methods. It contains metrics, and wrappers for datasets and methods.

## Installation
Run
```bash
pip install cpas_toolbox
```
to install the latest release of the toolbox. There is no need to download any additional weights or datasets. Upon first usage the evaluation script will ask to download the weights if they are not available at the expected path.

## Evaluation of baseline methods
To reproduce the REAL275 benchmark run:
```bash
python -m cpas_toolbox.evaluate --config real275.yaml --out_dir ./results/
```
To reproduce the REDWOOD75 benchmark run:
```bash
python -m cpas_toolbox.evaluate --config redwood75.yaml --out_dir ./results/
```

We can overwrite settings of the configuration via the command-line. For example, 
```bash
python -m cpas_toolbox.evaluate --config redwood75.yaml --out_dir ./results/ --visualize_gt True --visualize_prediction True
```
enables interactive visualization of ground truth and predictions. Alternatively, you could specify `--store_visualization True` to save the visualization of every prediction in the results directory.

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
