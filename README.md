# CS-SwinGAN for multi-coil MRI reconstruction

Official PyTorch implementation of CS-SwinGAN for multi-coil MRI reconstruction described in the paper "CS-SwinGAN: a swin-transformer-based generative adversarial network with compressed sensing pre-enhancement for multi-coil MRI reconstruction".
<br />
<br />
DOI: https://doi.org/10.1016/j.bspc.2025.108120
<br />
<br />
*Refreshed Pre-procoessing for Multi-coil Raw k-space data
<br />
<br />
*Channel Merging Reconstruction

<br />
<br />

<div align="center">
  <figure> 
    <img src="./asserts/preprocessing_block.png" width="800px">
    <figcaption><br />Pre-processing Block for Multi-coil MRI Raw Data</figcaption> 
  </figure> 
</div>

<br />
<br />

<div align="center">
  <figure> 
    <img src="./asserts/framework.png" width="800px">
    <figcaption><br />Full Framework of CS-SwinGAN for Multi-coil MRI Reconstruction</figcaption> 
  </figure> 
</div>

<br />
<br />

<div align="center">
  <figure> 
    <img src="./asserts/CS-Block.png" width="800px">
    <figcaption><br />Proposed Compressed Sensing Block</figcaption> 
    <figure> 
</div>

<br />
<br />

## Dependencies

```
easydict==1.10
einops==0.8.0
focal-frequency-loss==0.3.0
h5py==3.9.0
ipython==8.14.0
matplotlib==3.7.2
nibabel==5.1.0
numpy==1.25.2
opencv-python==4.8.0.74
Pillow==10.0.0
PyWavelets==1.4.1
PyYAML==6.0.1
scikit-image==0.21.0
scipy==1.11.1
# Editable install with no version control (setuptools==68.0.0)
timm==0.9.2
torch==2.0.1
tqdm==4.65.0
```

## Installation
- Clone this repo:
```bash
git clone https://github.com/notmayday/CS-SwinGAN_MC_Rec
cd CS-SwinGAN_MC_Rec
```

## Train

<br />

```
python3 train.py 

```


## Test

<br />

```
python3 difference_hot.py 

```
<br />
<br />

## Trained checkpoint download

We have established a checkpoint based on our ongoing work. For optimal results, we recommend training your own CS-SwinGAN_MC_Rec model.
<br />
[Brain_T2_cartesian_10%](https://drive.google.com/file/d/1vCrBJbypJ3mpEsYFNKTZTYuf5dNDmUbZ/view?usp=drive_link)
<br />
[Brain_T2_cartesian_20%](https://drive.google.com/file/d/1O1WrO5eypboHumXVGQ4eSgQRQaheoR_g/view?usp=drive_link)
<br />
[Brain_T2_cartesian_30%](https://drive.google.com/file/d/1MNvC1cjPBbpvsHGVyU7y6U3rUlKQLD3s/view?usp=drive_link)


# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.


<br />
@article{ZHANG2025108120,
title = {CS-SwinGAN: A swin-transformer-based generative adversarial network with compressed sensing pre-enhancement for multi-coil MRI reconstruction},
journal = {Biomedical Signal Processing and Control},
volume = {110},
pages = {108120},
year = {2025},
issn = {1746-8094},
doi = {https://doi.org/10.1016/j.bspc.2025.108120},
url = {https://www.sciencedirect.com/science/article/pii/S1746809425006317},
author = {Haikang Zhang and Zongqi Li and Qingming Huang and Luying Huang and Yicheng Huang and Wentao Wang and Bing Shen},
keywords = {Multi-coil MRI reconstruction, Deep learning, Loss separation, K-space noise suppression, Transformer},
abstract = {Magnetic resonance imaging (MRI) reconstruction from undersampled k-space data is a crucial area of research due to its potential to reduce scan times. Current deep learning approaches for MRI reconstruction often combine frequency-domain and image-domain losses, optimizing their sum. However, this approach can lead to blurry results, as it averages two fundamentally different types of losses. To address this issue, we propose CS-SwinGAN for multi-coil MRI reconstruction, a swin-transformer-based generative adversarial network with a Compressed Sensing Block for pre-enhancement. The newly introduced Compressed Sensing Block not only facilitates the separation of frequency-domain and image-domain losses but also serves as a pre-enhancement stage that promotes sparsity and suppresses aliasing, thereby enhancing reconstruction quality. We evaluate CS-SwinGAN in both standard MRI reconstruction tasks and under varying noise levels in k-space to assess its performance across diverse conditions. Numerical experiments demonstrate that our framework outperforms state-of-the-art methods in both conventional reconstruction and noise suppression scenarios. The source code is available at https://github.com/notmayday/CS-SwinGAN_MC_Rec.}
}

<br />

# Acknowledgements

This code uses libraries from [Subsampled-Brain-MRI-Reconstruction-by-Generative-Adversarial-Neural-Networks](https://github.com/ItamarDavid/Subsampled-Brain-MRI-Reconstruction-by-Generative-Adversarial-Neural-Networks) and [SwinGAN](https://github.com/learnerzx/SwinGAN) repositories.
# CS-SwinGAN for multi-coil MRI reconstruction
