import getopt
import math
import os
import sys

import cv2
import torch
import torchvision.transforms as transforms

from utils.data_loader import Dataset
from utils.pipeline import Pipeline


torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
print('Number of threads used: ', torch.get_num_threads())


os.environ['CUDA_HOME'] = '/opt/cuda/cuda-10.1'

input_path = 'images/doublestrike.jpg'
output_path = 'images/kbe'
dolly = False
output_frames = False
pretrained_estim = False
pretrained_refine = False
inpaint_depth = False
d2 = False

startU, startV = None, None
startW, startH = None, None
endU, endV = None, None
endW, endH = None, None


inpaint_path = './models/trained/inpainting-color.tar'
refine_path = './models/trained/disparity-refinement.tar'
estim_path = './models/trained/disparity-estimation-no-mask.tar'
inpaint_depth_path = './models/trained/inpainting-depth.tar' 

strParameter = ['in=', 'out=', 'dolly', 'write-frames', 'inpaint-path=', 
                'refine-path=', 'estim-path=', 'startU=', 'startV=', 'endU=', 
                'endV=', 'startW=', 'startH=', 'endW=', 'endH=', 'pretrained-refine', 'pretrained-estim', 'inpaint-depth=', '2d']

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', strParameter)[0]:
    if strOption == '--in' and strArgument != '': 
        input_path = strArgument # path to the input image
    if strOption == '--out' and strArgument != '': 
        output_path = strArgument # path to where the output should be stored

    if strOption == '--dolly': 
        dolly = True # perform dolly effect or not
    if strOption == '--write-frames': 
        output_frames = True # perform dolly effect or not
    if strOption == '--pretrained-refine': 
        pretrained_refine = True # if pretrained network from 3D KBE paper are used
    if strOption == '--pretrained-estim': 
        pretrained_estim = True # if pretrained network from 3D KBE paper are used
    if strOption == '--2d': 
        d2 = True # make a 2D KBE instead of 3D


    if strOption == '--inpaint-depth' and strArgument != '': 
        inpaint_depth = True # if pretrained network from 3D KBE paper are used
        inpaint_depth_path = strArgument
    if strOption == '--inpaint-path' and strArgument != '': 
        inpaint_path = strArgument # path to where the inpainting network is stored
    if strOption == '--refine-path' and strArgument != '': 
        refine_path = strArgument # path to where the refinement network is stored
    if strOption == '--estim-path' and strArgument != '': 
        estim_path = strArgument # path to where the estimation network is stored
    
    if strOption == '--startU' and strArgument != '': 
        startU = int(strArgument) 
    if strOption == '--startV' and strArgument != '': 
        startV = int(strArgument) 
    if strOption == '--endU' and strArgument != '': 
        endU = int(strArgument) 
    if strOption == '--endV' and strArgument != '': 
        endV = int(strArgument)

    if strOption == '--startW' and strArgument != '': 
        startW = int(strArgument) 
    if strOption == '--startH' and strArgument != '': 
        startH = int(strArgument) 
    if strOption == '--endW' and strArgument != '': 
        endW = int(strArgument) 
    if strOption == '--endH' and strArgument != '': 
        endH = int(strArgument) 
    
# end

if __name__ == '__main__':

    numpyImage = cv2.imread(filename=input_path, flags=cv2.IMREAD_COLOR)
    if pretrained_estim:
        numpyImage = cv2.cvtColor(numpyImage, cv2.COLOR_BGR2RGB)
    # tensorImage = torch.from_numpy(numpyImage).float()
    
    image_preparation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

    tensorImage = image_preparation(numpyImage)

    imgHeight = tensorImage.size()[1]
    imgWidth = tensorImage.size()[2]

    # Images need to have height and width multiple of 4, cropping a few pixel if it is not the case
    if imgWidth % 4 != 0:
        tensorImage = tensorImage[:,:, :-(imgWidth % 4)]
        imgWidth = tensorImage.size()[2]
    if imgHeight%4 != 0:
        tensorImage = tensorImage[:,:-(imgHeight % 4), :]
        imgHeight = tensorImage.size()[1]
    

    # if only one dimension is specified use input image aspect ratio
    if endH is not None and endW is None:
        endW = int(imgWidth * endH / imgHeight)
    if endW is not None and endH is None:
        endH = int(imgHeight * endW / imgWidth)
    
    if startH is not None and startW is None:
        startW = int(imgWidth * startH / imgHeight)
    if startW is not None and startH is None:
        startH = int(imgHeight * startW / imgWidth)

    if None in [startU, startV, startW, startH, endU, endV, endW, endH] and not dolly:
        print('At least one of the cropping parameters was not defined, using default ones for 3D kbe.')
        startU, startV = imgWidth / 2.15, imgHeight / 2.15
        startW, startH = int(math.floor(0.90 * imgWidth)), int(math.floor(0.90 * imgHeight))
        endU, endV = imgWidth / 1.85, imgHeight / 1.85
        endW, endH = int(math.floor(0.85 * imgWidth)), int(math.floor(0.85 * imgHeight))

    elif None in [startU, startV, startW, startH, endU, endV, endW, endH] and dolly:
        print('At least one of the cropping parameters was not defined, using default ones for dolly effect.')
        startU, startV = imgWidth / 2, imgHeight / 2
        startW, startH = int(math.floor(0.8 * imgWidth)), int(math.floor(0.8 * imgHeight))
        endU, endV = imgWidth / 2, imgHeight / 2
        endW, endH = int(math.floor(0.3 * imgWidth)), int(math.floor(0.3 * imgHeight))
    
    assert imgHeight >= startV + startH/2 and startV - startH/ 2>= 0, 'Start window too tall compared to given center'
    assert imgWidth >= startU + startW/2 and startU - startW/ 2>= 0, 'Start window too tall compared to given center'

    assert imgHeight >= endV + endH/2 and endV - endH/ 2>= 0, 'End window too tall compared to given center'
    assert imgWidth >= endU + endW/2 and endU - endW/ 2>= 0, 'End window too tall compared to given center'

    # assert endH / endW == startH / startW, 'Starting and ending aspect ratio are different'

    tensorImage = tensorImage.view(1, 3, imgHeight, imgWidth)

    objectFrom = {
        'dblCenterU': startU,
        'dblCenterV': startV,
        'intCropWidth': startW,
        'intCropHeight': startH
    }

    objectTo = {
        'dblCenterU': endU,
        'dblCenterV': endV,
        'intCropWidth': endW,
        'intCropHeight': endH
    }

    zoom_settings = {
        'objectFrom' : objectFrom,
        'objectTo' : objectTo
    }

    if inpaint_depth:
        ken_burn_pipe = Pipeline(model_paths=[estim_path, refine_path, inpaint_path, inpaint_depth_path], 
                                dolly=dolly, output_frames=output_frames, pretrain=pretrained_refine, d2=d2)
    else:
        ken_burn_pipe = Pipeline(model_paths=[estim_path, refine_path, inpaint_path], 
                                dolly=dolly, output_frames=output_frames, pretrain=pretrained_refine, d2=d2)


    # ken_burn_pipe = Pipeline()
    with torch.no_grad():
        ken_burn_pipe((tensorImage+1)/2, zoom_settings, output_path, pretrained_estim=pretrained_estim)
