# ScribbleCont

This is official pytorch implementation of SCRIBBLE-SUPERVISED CELL SEGMENTATION USING MULTISCALE CONTRASTIVE REGULARIZATION (ISBI 2022).
![Framework](./figure/framework.pdf)

## Setup
This work has been implemented on Ubuntu 20.04, Python 3.7, pytorch 1.7.1, CUDA 11.0 and GeForce RTX 3090 GPU.
* Install python packages 
```
pip install -r requirements.txt
```
* Dataset (BBBC038v1)
   * Download dataset from [official website](https://bbbc.broadinstitute.org/BBBC038)
   * Preprocess code here [scribble2label](https://github.com/hvcl/scribble2label)
```
/bin/bash preprocess_dataset.sh
```
* Dataset (MoNuSeg)
   * Download dataset from [official website](https://monuseg.grand-challenge.org/Data/)
   * Preprocess the dataset

## Implementation
* Training
```
python main.py
```
* Inference
```
python inference.py
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
