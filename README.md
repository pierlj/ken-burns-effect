# Improved 3D Ken Burns effect

Development repository for Pierre Le Jeune's master thesis at DTU.

This repository contains code for generating 3D Ken Burns effect from single image. This work is mainly based on [3D Ken Burns Effect from a Single Image(]https://arxiv.org/pdf/1909.05483.pdf). This paper was a starting for our work and therefore the pre-trained networks released by Niklaus et al. are comptible with our framework and can be downloaded from their [repo](https://github.com/sniklaus/3d-ken-burns).  

We provide code for the training of the different neural network used to achieve the 3D Ken Burns effect. In addition we propose some extension of the original work, to improve both the depth estimation and the image inpainting. Finally, we develop an semi-supervised method for the disocclusion inpainting problem in order to prevent the difficulty of getting a synthetic dataset as used in the original paper. We also proposed a slight modification of the 3D KBE to produce fake [dolly zoom](https://en.wikipedia.org/wiki/Dolly_zoom). 

## Generate 3D Ken Burns effects
GIF HERE

## Synthetic Dolly zoom
GIF HERE 

## Training notes
In order to train the networks, one must set different parameters that are modifiable in the file `main.py`.
* A dataset must be specified, it should contain the pairs (image, depth). To define a dataset just create a `dict` object in the main file. This dict must contains at least three attributes:
  * `name`: the name of the dataset (only for display).
  * `path`: the path to the files. In this folder there should be two folders `images` and `depth` which contain respectively the RGB images and associated depth maps. Correspong image and depth should have the same name. 
  * `param`: the camera parameters used for that specific dataset. It is also a `dict` with only two attributes `baseline` and `focal`, the baseline and focal length of the camera. 
 
Note that some adjustment may be required in the data loading file (`data_loader.py`) in order to match the folder layout and files extension of the images and depth maps. 

* Then a trainer class instance must be created, there exists one for depth estimation training and one for inpainting. Both are called the same way, they require 3 parameters:
  * A dataset `list` where the elements are the dataset `dict` defined above.
  * A `dict` of training parameters (see below).
  * (optional) a `list` of files that contain pre-trained nets. 
  
### Depth estimation training


### Inpainting training

**Important notes** 
- It is important to set properly the `CUDA_HOME` variable according to your system so that CUDA kernels can be executed.
- It is highly recommended to run these on GPU. 
