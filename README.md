# ScribbleCont

This is official pytorch implementation of SCRIBBLE-SUPERVISED CELL SEGMENTATION USING MULTISCALE CONTRASTIVE REGULARIZATION (ISBI 2022).

## Setup
This work has been implemented on Ubuntu 20.04, Python 3.7, pytorch 1.7.1, CUDA 11.0 and GeForce RTX 3090 GPU.
* Install python packages 
```
pip install -r requirements.txt
```
* Dataset (BBBC038v1)
    <br> * Download the dataset from [the official website](https://bbbc.broadinstitute.org/BBBC038)
    <br> * Preprocess code here [scribble2label](https://github.com/hvcl/scribble2label)
* Dataset (MoNuSeg)
```
```

## Citation
Please cite us if you use our work:
```
@inproceedings{hj_isbi22,
title={Scribble-supervised Cell Segmentation Using Multiscale Contrastive Regularization},
author={Oh, Hyun-Jic and Lee, Kanggeun and Jeong, Won-Ki},
booktitle={2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI)},
pages={},
year={2022},
organization={IEEE}
}
```
