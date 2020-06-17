# Improved 3D Ken Burns effect

This repository contains code for generating 3D Ken Burns effect from single image. This work is mainly based on [3D Ken Burns Effect from a Single Image](https://arxiv.org/pdf/1909.05483.pdf). This paper was a starting point for our work and therefore the trained networks released by Niklaus et al. are comptible with our framework and can be downloaded from their [repository](https://github.com/sniklaus/3d-ken-burns).  

We provide code for the training of the different neural networks used to achieve the 3D Ken Burns effect. In addition we propose some extension of the original work, to improve both the depth estimation and the image inpainting. Finally, we develop an semi-supervised method for the disocclusion inpainting problem in order to prevent the difficulty of getting a synthetic dataset as used in the original paper. We also proposed a slight modification of the 3D KBE to produce fake [dolly zoom](https://en.wikipedia.org/wiki/Dolly_zoom). 

Here is a [video](https://www.youtube.com/watch?v=nSZrJOJFj9o) to some of our results.

## [Generate 3D Ken Burns effects](https://www.youtube.com/watch?v=nSZrJOJFj9o)
![](https://github.com/ipeter50/ken-burns-effect/blob/master/images/3D_Ken_Burns_effect.gif)

To generate 3D KBE use the script `kbe.py`. Some parameters can be set to change from default settings. If no path for the networks are specified, default names and paths from donwload script will be used.
* Selection of the input image and the networks to be used:
    * `in=`: path to input image
    * `out=`: saving path
    * `inpaint-path=`: path to inpainting network              
    * `refine-path=`: path to refinement network
    * `estim-path=`: path to estimation network
    * `inpaint-depth=`: use different network for color and depth inpainting
    * `pretrained-refine`: must be set when using trained refinement network from original paper
    * `pretrained-estim`: must be set when using trained estimation network from original paper

* Specifying the cropping windows:
    * `startU=`: x coordinate of the starting crop window
    * `startV=`: y coordinate of the starting crop window
    * `endU=`: x coordinate of the ending crop window 
    * `endV=`: y coordinate of the ending crop window
    * `startW=`: width of the starting crop window
    * `startH=`: height of the starting crop window
    * `endW=`: width of the ending crop window
    * `endH=`: height of the ending crop window

If some of the cropping windows parameters are not specified, default parameters will be applied. 

This will create a video of the 3D KBE but frames of that video can be outputed as well with option `--wrtie-frames`

Example:
```
CUDA_AVAILABLE_DEVICES=X python kbe.py --in /images/test.png --out /images/kbe/ --estim-path /models/trained/disparity-estimation.tar --refine-path /models/trained/disparity-refinement.tar --inpaint-path /models/trained/inpainting-color.tar --write-frames --startU 512 --startV 512 --endU 600 --endV 600 --startW 400 --startH 200 --endW 300 --endH 150 
```

## [Synthetic Dolly zoom](https://www.youtube.com/watch?v=wC1_anb8eHw)
![](https://github.com/ipeter50/ken-burns-effect/blob/master/images/Dolly_zoom_effect_from_single_image.gif)

To create dolly zoom effect use the `--dolly` option with the `kbe.py` script. It will work exactly as for the 3D KBE except that the focal length of the camera will change during the effect to compensate the forward or backward motion, keeping the focused object unchanged. It is recommended to set the same position for the centers of the two cropping windows in order to remove any lateral motion. 

Example:
```
CUDA_AVAILABLE_DEVICES=X python kbe.py --in /images/test.png --out /images/kbe/ --estim-path /models/trained/disparity-estimation.tar --refine-path /models/trained/disparity-refinement.tar --inpaint-path /models/trained/inpainting-color.tar --write-frames --startU 512 --startV 512 --endU 512 --endV 512 --startW 400 --startH 200 --endW 300 --endH 150 --dolly
```


## Training notes
In order to train the network, the script `train.py` can be used.
```
CUDA_AVAILABLE_DEVICES=X python train.py
```
A few  parameters are available in order to control the training:
* `training-mode`: select which network to train, can take values: `estimation`, `refinement`, `inpainting`, `inpainting_ref`
* `mask-loss`: choose the type of mask loss, can take values `none` (no mask loss), `same` (mask loss computed on depth dataset) or `other` (mask loss computed on another dataset)
* `mask-loss-dataset`: path to dataset to be used for mask loss, required when `mask-loss=other`
* `n-epochs`
* `lr-estimation`: learning rate for estimation net
* `lr-refinement`: learning rate for refinement net
* `lr-inpaint`: learning rate for inpainting net
* `lr-discriminator`: learning rate for discrimintor
* `save-name`: name for saving network weights
* `model-path`: path to pre-trained network for refinement or continue training
* `batch-size`
* `gamma-lr`: learning rate decay rate
* `partial-conv`: use partial conv, only for inpainting 

Example for training the inpainting network:
```
CUDA_AVAILABLE_DEVICES=X python train.py --training-mode inpainting --batch-size=2 --lr-inpaint 0.0005 --save-name test --partial-conv 
```
Then a dataset must be specified, it should contain the pairs (image, depth). To define a dataset just create a `dict` object in the file `train.py`. This dict must contains at least three attributes:
  * `name`: the name of the dataset (only for display).
  * `path`: the path to the files. In this folder there should be two folders `images` and `depth` which contain respectively the RGB images and associated depth maps. Correspong image and depth should have the same name. 
  * `param`: the camera parameters used for that specific dataset. It is also a `dict` with only two attributes `baseline` and `focal`, the baseline and focal length of the camera. 
  
Multiple datasets can be used for training, simply feed a dataset list to the `Trainer` class.
 
Note that some adjustment may also be required in the data loading file (`data_loader.py`) in order to match the folder layout and files extension of the images and depth maps. 

**Important notes** 
- It is important to set properly the `CUDA_HOME` variable according to your system so that CUDA kernels can be executed.
- It is highly recommended to run these scripts on a GPU. Minimum recommended memory would be 6 GB.
- Dependencies: `PyTorch=1.3.1`, `kornia=0.3.0`, `cupy`, `h5py`, `opencv`, `dill`.
