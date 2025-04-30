# Learning Differentiable Tensegrity Dynamics with Graph Neural Networks

Code for the paper: https://arxiv.org/abs/2410.12216

## Install conda

## Create virtual env

``conda create --name tensegrity_gnn python=3.10
conda activate tensegrity_gnn
pip install -r requirements.txt
``

## Download dataset

https://drive.google.com/file/d/1zQI3JC8I2l1BWdmuTbFnYIcTCEW7MY_Y/view?usp=sharing

## Prepare data

```angular2html
cd ../
mkdir tensegrity
cd tensegrity
mkdir models data_sets
cd data_sets
mv {DOWNLOAD_DIR}/mjc_synthetic_5d_0.01.zip ./
unzip mjc_synthetic_5d_0.01.zip
```

## Train Model

``python3 train.py``

## Eval model

``python3 eval.py``

## Modifying config files for your own experiments

There are two config json files relevant for simulation and training:
1. `training/configs/*`    <--- json files that control training related parameters/settings
2. `simulators/configs/*`   <--- json files that specify simulation and robot parameters and configuration


## Miscellaneous

- Default tensor precision is float32. Although this is often good enough and train/run faster, 
switch `DEFAULT_DTYPE` to float64 in `utilties/misc_utils` for more accuracy and stability.

## If this work was useful for your research, please cite:

```angular2html
@misc{chen2024learningdifferentiabletensegritydynamics,
      title={Learning Differentiable Tensegrity Dynamics using Graph Neural Networks}, 
      author={Nelson Chen and Kun Wang and William R. Johnson III au2 and Rebecca Kramer-Bottiglio and Kostas Bekris and Mridul Aanjaneya},
      year={2024},
      eprint={2410.12216},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.12216}, 
}
```

