# Data preparation

On first usage of a dataset the script will download and preprocess the datasets automatically. This is the recommended way to use the package as it ensures an unmodified dataset.

If you already downloaded a dataset and want to use symlinks instead of storing them again to save storage space, you can follow the manual instructions below.

## REAL275
For download links check the [NOCS repository](https://github.com/hughw19/NOCS_CVPR2019).

The expected folder structure for REAL275 evaluation is as follows:
```
    {root_dir}/real_test/...
    {root_dir}/gts/...
    {root_dir}/obj_models/...
```
An additional directory `{root_dir}/csap_toolbox/` will be created to store preprocessed files. By default `{root_dir}` will be `data/nocs/` (i.e., relative to the current working directory, when executing the evaluation script), but it can be modified.

## REDWOOD75
