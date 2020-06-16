import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from models.disparity_estimation import Disparity, Semantics
from models.disparity_refinement import Refine
from utils.data_loader import Dataset
from utils.utils import (CustomWriter, compute_metrics, device, load_models,
                         normalize_torch_tensor, resize_image)


class DepthEval():
    def __init__(self, dataset_paths, model_paths, eval_refine=False, eval_pretrained=False, perform_adjustment=True, evaluation_path=None, imagenet_path=None):
        self.dataset = Dataset(dataset_paths, imagenet_path=imagenet_path)
        self.dataset.mode = 'eval'
        self.dataset.padding = True
        self.data_loader = self.dataset.get_dataloader(batch_size=1)
        self.perform_adjustment = perform_adjustment
        self.dataset_paths = dataset_paths

        torch.manual_seed(42)
        np.random.seed(42)

        self.moduleSemantics = Semantics().to(device).eval()
        self.moduleDisparity = Disparity().to(device).eval()
        self.moduleMaskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()
        self.moduleRefine = Refine().to(device).eval()

        self.eval_pretrained = eval_pretrained

        if model_paths is not None:
            models_list = [{'model':self.moduleDisparity,
                                'type':'disparity'},
                            {'model':self.moduleRefine,
                                'type':'refine'}]
            load_models(models_list, model_paths)

    
    def eval(self):
        # compute the metrics on the provided dataset with the provided networks
        measures = []

        metrics = {}
        metrics_list = ['Abs rel', 'Sq rel', 'RMSE', 'log RMSE', 's1', 's2', 's3']
        MSELoss = nn.MSELoss()

        print('Starting evaluation on datasets: ', functools.reduce(lambda s1, s2: s1['path'] + ', ' + s2['path'], self.dataset_paths))

        for idx, (tensorImage, disparities, masks, imageNetTensor, dataset_ids) in enumerate(tqdm(self.data_loader)):
            tensorImage = tensorImage.to(device, non_blocking=True)
            disparities = disparities.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            N = tensorImage.size()[2]*tensorImage.size()[3]
            
            # pretrained networks from 3D KBE were trained with image normalized between 0 and 1
            if self.eval_pretrained:
                tensorImage = (tensorImage + 1) / 2

            tensorResized = resize_image(tensorImage)

            tensorDisparity = self.moduleDisparity(tensorResized, self.moduleSemantics(tensorResized))  # depth estimation
            tensorDisparity = self.moduleRefine(tensorImage, tensorDisparity) # increase resolution
            tensorDisparity = F.threshold(tensorDisparity, threshold=0.0, value=0.0)

            masks = masks.clamp(0,1)
            measures.append(np.array(compute_metrics(tensorDisparity, disparities, masks)))
    
        
        measures = np.array(measures).mean(axis=0)

        for i, name in enumerate(metrics_list):
            metrics[name] = measures[i]
        
        return metrics
    
    def get_depths(self):
        # return input images and predictions 
        def detach_tensor(tensor):
            return tensor.cpu().detach().numpy()

        tensorImage, disparities, masks, imageNetTensor, dataset_ids= next(iter(self.data_loader))
        tensorImage = tensorImage.to(device, non_blocking=True)
        disparities = disparities.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # pretrained networks from 3D KBE were trained with image normalized between 0 and 1
        if self.eval_pretrained:
            tensorImage = (tensorImage + 1) / 2

        tensorResized = resize_image(tensorImage)


        # retrieve parameters for different sets of images
        tensorFocal = torch.Tensor([self.dataset_paths[int(id.item())]['params']['focal'] for id in dataset_ids])
        tensorBaseline = torch.Tensor([self.dataset_paths[int(id.item())]['params']['baseline'] for id in dataset_ids])
        tensorFocal = tensorFocal.view(-1,1).repeat(1, 1, tensorImage.size(2) * tensorImage.size(3)).view(*disparities.size())
        tensorBaseline = tensorBaseline.view(-1,1).repeat(1, 1, tensorImage.size(2) * tensorImage.size(3)).view(*disparities.size())

        tensorBaseline = tensorBaseline.to(device)
        tensorFocal = tensorFocal.to(device)

        tensorDisparity = self.moduleDisparity(tensorResized, self.moduleSemantics(tensorResized))  # depth estimation

        objectPredictions = self.moduleMaskrcnn(tensorImage) # segment image in mask using Mask-RCNN

        tensorDisparityAdjusted = tensorDisparity
        tensorDisparityRefined = self.moduleRefine(tensorImage[:2,:,:,:], tensorDisparityAdjusted[:2,:,:,:]) # increase resolution

        return (detach_tensor(tensorDisparity), 
                detach_tensor(tensorDisparityAdjusted), 
                detach_tensor(tensorDisparityRefined), 
                detach_tensor(disparities),
                detach_tensor(resize_image(disparities, max_size=256)),
                detach_tensor((tensorImage.permute(0,2,3,1)+1)/2),
                objectPredictions,
                detach_tensor(masks),
                detach_tensor(resize_image(masks, max_size=256)))
