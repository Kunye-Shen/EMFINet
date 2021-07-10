# EMFINet (TGRS 2021)
'[Edge-Aware Multiscale Feature Integration Network for Salient Object Detection in Optical Remote Sensing Images](https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=9474908&ref=aHR0cHM6Ly9pZWVleHBsb3JlLmllZWUub3JnL2Fic3RyYWN0L2RvY3VtZW50Lzk0NzQ5MDg=)', [Xiaofei Zhou](https://scholar.google.com.hk/citations?hl=zh-CN&user=2PUAFW8AAAAJ), [Kunye Shen](https://scholar.google.com.hk/citations?hl=zh-CN&user=q6_PkywAAAAJ), [Zhi Liu](https://scholar.google.com.hk/citations?hl=zh-CN&user=Sd5VB2cAAAAJ), [Chen Gong](https://scholar.google.com.hk/citations?user=guttoBwAAAAJ&hl=zh-CN), Jiyong Zhang, and Chenggang Yan.

## Required libraries

Python 3.7  
numpy 1.18.1  
scikit-image 0.16.2  
PyTorch 1.4.0  
torchvision 0.5.0  
glob  

The SSIM loss is adapted from [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py).

## Architecture
![EMFINet architecture](figures/architecture.png)

## Quantitative Comparison
![Quantitative Comparison](figures/quan.png)

## Qualitative Comparison
### ORSSD
![ORSSD](figures/qual_ORSSD.png)

### EORSSD
![EORSSD](figures/qual_EORSSD.png)

## Citation
```
@InProceedings{Qin_2019_CVPR,
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Gao, Chao and Dehghan, Masood and Jagersand, Martin},
title = {BASNet: Boundary-Aware Salient Object Detection},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
