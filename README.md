<div align="center">

# Batteries Included PyTorch Tutorial

[![python](https://img.shields.io/badge/-Python_3.9-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.5.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

This is a basic PyTorch tutorial. To get started, simply git clone this repo and cd into the folder:
```
git clone https://github.com/SciPritchardLab/pytorch_tutorial.git
cd pytorch_tutorial
```
Then, install and activate the conda environment.
```
conda env create -f environment.yaml
conda activate tutorialenv
```

The environment.yml assumes you are using MacOS. A pull request adding environments for Linux and Windows would be welcomed.

To play with the marimo notebook simply enter:

```
marimo edit train_marimo.py
```

To train models and iterate from the command line, simply enter:

```
python train.py --config-name=config_default
```

You can run additional experiments simply by modifying the config file or creating and using new config files.

This repo accompanies slides found here:

https://docs.google.com/presentation/d/1DdT6Eud3avT4K-5UCgyNKGxF5ZIqAuIaJ_-74hpAxf0/edit?usp=sharing
