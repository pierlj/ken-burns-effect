import math
import os

import cv2
import imageio as io
import moviepy
import moviepy.editor
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from models.disparity_estimation import Disparity, Semantics
from models.disparity_refinement import Refine
from models.disparity_refinement_pretrained import Refine as RefineP
from models.partial_inpainting import Inpaint as PartialInpaint
from models.pointcloud_inpainting import Inpaint
from utils.common import depth_to_points, process_kenburns
from utils.utils import device, load_models, resize_image


class Pipeline():
    def __init__(self, model_paths=None, partial_inpainting=False, dolly=False, output_frames=False, pretrain=False, d2=False):
        self.objectCommon = {}
        self.objectCommon['dblFocal'] = 1024.0/2
        self.objectCommon['dblBaseline'] = 120

        self.partial_inpainting = partial_inpainting
        self.dolly = dolly
        self.output_frames = output_frames
        self.d2 = d2

        self.moduleSemantics = Semantics().to(device).eval()
        self.moduleDisparity = Disparity().to(device).eval()
        self.moduleMaskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()
        if pretrain:
            self.moduleRefine = RefineP().to(device).eval()
        else:
            self.moduleRefine = Refine().to(device).eval()

        if self.partial_inpainting:
            self.moduleInpaint = PartialInpaint().to(device).eval()
        else:
            self.moduleInpaint = Inpaint().to(device).eval()
        
        models_list = [{'model':self.moduleDisparity,
                            'type':'disparity'},
                        {'model':self.moduleRefine,
                            'type':'refine'},
                        {'model':self.moduleInpaint,
                            'type':'inpaint'}]
        if len(model_paths) == 4:
            self.moduleInpaintDepth = Inpaint().to(device).eval()
            models_list.append({'model':self.moduleInpaintDepth, 'type':'inpaint'})

        load_models(models_list, model_paths)

    def __call__(self, tensorImage, zoom_settings, output_path=None, inpaint_depth=False, pretrained_estim=False):

        tensorImage = tensorImage.to(device)

        # tensorImage should be (1xCxHxW) only one image is feeded at a time through the pipeline
        self.objectCommon['intWidth'] = tensorImage.size()[3]
        self.objectCommon['intHeight'] = tensorImage.size()[2]

        self.objectCommon['tensorRawImage'] = tensorImage


        tensorImage = tensorImage.contiguous()

        tensorResized = resize_image(tensorImage, max_size=int(max(self.objectCommon['intWidth'], self.objectCommon['intHeight'])/2))

        tensorDisparity = self.moduleDisparity(tensorResized, self.moduleSemantics(tensorResized))  # depth estimation
        if self.d2:
            tensorDisparity = torch.ones_like(tensorDisparity)

        tensorDisparity = self.moduleRefine(tensorImage, tensorDisparity) # increase resolution
        if tensorDisparity.min() < 0.0:
            tensorDisparity -= tensorDisparity.min()
        tensorDisparity = tensorDisparity / tensorDisparity.max() * self.objectCommon['dblBaseline']  # normalize disparities

        # Create 3D model from disparity, via depth
        tensorDepth = (self.objectCommon['dblFocal'] * self.objectCommon['dblBaseline']) / (tensorDisparity + 1e-7)
        tensorPoints = depth_to_points(tensorDepth, self.objectCommon['dblFocal'])

        # Delete networks variable to free memory as they are no longer to be used 
        del self.moduleSemantics
        del self.moduleDisparity
        del self.moduleMaskrcnn
        del self.moduleRefine

        # Store useful data for next steps.
        self.objectCommon['dblDispmin'] = tensorDisparity.min().item()
        self.objectCommon['dblDispmax'] = tensorDisparity.max().item()
        self.objectCommon['objectDepthrange'] = cv2.minMaxLoc(src=tensorDepth[0, 0, 128:-128, 128:-128].detach().cpu().numpy(), mask=None)
        self.objectCommon['tensorRawPoints'] = tensorPoints.view(1, 3, -1)
        self.objectCommon['tensorRawImage'] = tensorImage
        self.objectCommon['tensorRawDisparity'] = tensorDisparity
        self.objectCommon['tensorRawDepth'] = tensorDepth

        if inpaint_depth:
            numpyResult = process_kenburns({
            'dblSteps': np.linspace(0.0, 1.0, 75).tolist(),
            'objectFrom': zoom_settings['objectFrom'],
            'objectTo': zoom_settings['objectTo'],
            'boolInpaint': True,
            'dolly': self.dolly
        }, self.objectCommon, [self.moduleInpaint, self.moduleInpaintDepth])

        else:
            numpyResult = process_kenburns({
                'dblSteps': np.linspace(0.0, 1.0, 75).tolist(),
                'objectFrom': zoom_settings['objectFrom'],
                'objectTo': zoom_settings['objectTo'],
                'boolInpaint': True,
                'dolly': self.dolly
            }, self.objectCommon, self.moduleInpaint)

        if self.output_frames:
            for idx, frame in enumerate(tqdm(numpyResult, desc='Saving video frames')):
                frames_dir = os.path.join(output_path, 'frames')
                if not os.path.exists(frames_dir):
                    os.makedirs(frames_dir)
                if pretrained_estim:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(frames_dir + '/' +str(idx) + '.png', frame)

        # Create video output
        if output_path is not None:
            if pretrained_estim:
                moviepy.editor.ImageSequenceClip(sequence=[ numpyFrame[:, :, :] for numpyFrame in numpyResult + list(reversed(numpyResult))[1:] ], fps=25).write_videofile(os.path.join(output_path,'3d_kbe.mp4'), codec='mpeg4')
            else:
                moviepy.editor.ImageSequenceClip(sequence=[ numpyFrame[:, :, ::-1] for numpyFrame in numpyResult + list(reversed(numpyResult))[1:] ], fps=25).write_videofile(os.path.join(output_path,'3d_kbe.mp4'), codec='mpeg4')
