import functools

import dill
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from models.pointcloud_inpainting import Inpaint
from models.partial_inpainting import Inpaint as PartialInpaint

from utils.data_loader import Dataset, ImageNetDataset
from utils.utils import *
from utils.losses import *
from utils.common import spatial_filter, depth_to_points, generate_mask
from utils.fid import FID



class InpaintEval():
    def __init__(self, dataset_paths, models_paths=None, partial_conv=False):

        torch.manual_seed(543)
        np.random.seed(42)

        self.dataset_paths = dataset_paths

        self.dataset = Dataset(dataset_paths, mode='inpaint-eval')

        # Create training and validation set randomly
        self.dataset_length = len(self.dataset)


        if not partial_conv:
            self.moduleInpaint = Inpaint().to(device).train()
        else:
            self.moduleInpaint = PartialInpaint().to(device).train()

       
        if models_paths is not None:
            print('Loading model state from '+ str(models_paths))
            models_list = [{'model':self.moduleInpaint,'type':'inpaint'}]

            load_models(models_list, models_paths)

        

        self.data_loader = torch.utils.data.DataLoader(self.dataset, 
                                                        batch_size=1, 
                                                        shuffle=True, 
                                                        pin_memory=True, 
                                                        num_workers=1)

    def eval(self):
        camera = {'focal':1024/2.0, 'baseline':74.0}
        np.random.seed(42)
        self.fidNetwork = FID()
        measures = []

        metrics = {}
        # psnrImg, psnrDisp, ssimImg, ssimDisp, fid
        metrics_list = ['PSNR Image', 'PSNR Disparity', 'SSIM Image ', 'SSIM Disparity', 'FID']

        inception_activations_real = np.zeros((self.dataset_length, 2048), dtype=np.float32)
        inception_activations_inpaint = np.zeros((self.dataset_length, 2048), dtype=np.float32)

        print('Starting evaluation on datasets: ', functools.reduce(lambda s1, s2: s1['path'] + ', ' + s2['path'], self.dataset_paths))

        for idx, (tensorImage, tensorDisparity, tensorDepth, zoom_from, zoom_to, dataset_ids) in enumerate(tqdm(self.data_loader, desc='Evaluation')):
            # if idx > 9:
            #     break
            with torch.no_grad():
                batch_size = tensorImage.shape[0]
                start_idx = batch_size * idx
                end_idx = batch_size * (idx + 1)

                zoom_settings = {'objectFrom' : zoom_from, 'objectTo' : zoom_to}

                tensorImage = tensorImage.to(device, non_blocking=True)
                tensorDisparity = tensorDisparity.to(device, non_blocking=True)
                tensorDepth = tensorDepth.to(device, non_blocking=True)

                tensorMasks, tensorShift, objectList = get_masks(tensorImage, tensorDisparity, tensorDepth, zoom_settings, camera)

                # tensorImage, tensorDisparity = self.moduleInpaint.normalize_images_disp(tensorImage, tensorDisparity, not_normed=True)
                tensorImage = (tensorImage + 1) / 2
                # print(tensorImage.min(),tensorImage.max())

                inpaintObject = self.moduleInpaint(tensorImage=tensorImage * tensorMasks, 
                                                    tensorDisparity=tensorDisparity * tensorMasks, 
                                                    tensorMasks=tensorMasks)

                inpaintImage = inpaintObject['tensorImage']
                inpaintDisparity = inpaintObject['tensorDisparity']

                # tensorImage, tensorDisparity = self.moduleInpaint.normalize_images_disp(tensorImage, tensorDisparity, not_normed=False) 
                # print(inpaintImage.min(), inpaintImage.max())
                # print(tensorImage.min(),tensorImage.max())
                batch_metrics = compute_inpaint_metrics(inpaintImage, inpaintDisparity, tensorImage, tensorDisparity, tensorMasks)

                inpaintImage = torch.clamp(inpaintImage, 0.0, 1.0)

                # FID computation

                images_processed = self.fidNetwork.preprocess_images(tensorImage.cpu().permute(0,2,3,1).numpy(), False).to(device)
                images_inpaint_processed = self.fidNetwork.preprocess_images(inpaintImage.cpu().permute(0,2,3,1).numpy(), False).to(device)

                activations_real = self.fidNetwork.inception_network(images_processed).detach().cpu().numpy()
                assert activations_real.shape == (images_processed.shape[0], 2048)
                activations_inpaint =  self.fidNetwork.inception_network(images_inpaint_processed).detach().cpu().numpy()
                assert activations_inpaint.shape == (images_inpaint_processed.shape[0], 2048)

                inception_activations_real[start_idx:end_idx, :] = activations_real
                inception_activations_inpaint[start_idx:end_idx, :] = activations_inpaint
                
                

                measures.append(np.array(batch_metrics))
    
        
        measures = np.array(measures).mean(axis=0)

        mu_real, sigma_real = self.fidNetwork.calculate_activation_statistics(inception_activations_real)
        mu_inpaint, sigma_inpaint =  self.fidNetwork.calculate_activation_statistics(inception_activations_inpaint)

        fid = self.fidNetwork.calculate_frechet_distance(mu_real, sigma_real, mu_inpaint, sigma_inpaint)

        measures = np.append(measures, fid)

        for i, name in enumerate(metrics_list):
            metrics[name] = measures[i]
        
        return metrics
    
    def eval_adv(self):
        camera = {'focal':1024/2.0, 'baseline':74.0}
        np.random.seed(42)
        self.fidNetwork = FID()

        inception_activations_real = np.zeros((self.dataset_length, 2048), dtype=np.float32)
        inception_activations_inpaint = np.zeros((self.dataset_length, 2048), dtype=np.float32)

        print('Starting evaluation on datasets: ', functools.reduce(lambda s1, s2: s1['path'] + ', ' + s2['path'], self.dataset_paths))

        for idx, (tensorImage, tensorDisparity, tensorDepth, zoom_from, zoom_to, dataset_ids) in enumerate(tqdm(self.data_loader, desc='Evaluation')):
            # if idx > 9:
            #     breaknp.random.seed(42)
            with torch.no_grad():
                batch_size = tensorImage.shape[0]
                start_idx = batch_size * idx
                end_idx = batch_size * (idx + 1)

                zoom_settings = {'objectFrom' : zoom_from, 'objectTo' : zoom_to}
                tensorImage = tensorImage.to(device, non_blocking=True)
                tensorDisparity = tensorDisparity.to(device, non_blocking=True)
                tensorDepth = tensorDepth.to(device, non_blocking=True)


                tensorImage = (tensorImage + 1) / 2

                tensorImage, tensorDisparity = self.moduleInpaint.normalize_images_disp(tensorImage, tensorDisparity, not_normed=True)
                tensorContextA = self.moduleInpaint.moduleContext(torch.cat([ tensorImage, tensorDisparity ], 1))

                tensorRenderB, tensorMaskB, tensorPointsA, tensorShift, objectList = get_masks(tensorImage, tensorDisparity, tensorDepth, zoom_settings, camera,  
                                                                    AFromB=False, tensorContext=tensorContextA)

                tensorImageB, tensorDisparityB, tensorContextB = tensorRenderB[:,:3,:,:], tensorRenderB[:,3:4,:,:], tensorRenderB[:,4:,:,:]
                inpaintObjectB = self.moduleInpaint(tensorImage=tensorImageB,
                                                    tensorDisparity=tensorDisparityB, 
                                                    tensorMasks=tensorMaskB,
                                                    tensorContext=tensorContextB)

                inpaintImage = inpaintObjectB['tensorImage']
                inpaintDisparity = inpaintObjectB['tensorDisparity']   
                tensorImage, tensorDisparity = self.moduleInpaint.normalize_images_disp(tensorImage, tensorDisparity, not_normed=False)         
                # inpaintImage, inpaintDisparity = self.moduleInpaint.normalize_images_disp(inpaintImage, inpaintDisparity, not_normed=False)   

                tensorImage = torch.clamp(tensorImage, 0.0, 1.0)   
                inpaintImage = torch.clamp(inpaintImage, 0.0, 1.0)

                # FID computation

                images_processed = self.fidNetwork.preprocess_images(tensorImage.cpu().permute(0,2,3,1).numpy(), False).to(device)
                images_inpaint_processed = self.fidNetwork.preprocess_images(inpaintImage.cpu().permute(0,2,3,1).numpy(), False).to(device)

                activations_real = self.fidNetwork.inception_network(images_processed).detach().cpu().numpy()
                assert activations_real.shape == (images_processed.shape[0], 2048)
                activations_inpaint =  self.fidNetwork.inception_network(images_inpaint_processed).detach().cpu().numpy()
                assert activations_inpaint.shape == (images_inpaint_processed.shape[0], 2048)

                inception_activations_real[start_idx:end_idx, :] = activations_real
                inception_activations_inpaint[start_idx:end_idx, :] = activations_inpaint
                
    

        mu_real, sigma_real = self.fidNetwork.calculate_activation_statistics(inception_activations_real)
        mu_inpaint, sigma_inpaint =  self.fidNetwork.calculate_activation_statistics(inception_activations_inpaint)

        fid = self.fidNetwork.calculate_frechet_distance(mu_real, sigma_real, mu_inpaint, sigma_inpaint)

        return fid
    
    def get_inpaint(self, outputRenderC=False):
        
        camera = {'focal':1024/2.0, 'baseline':74.0}

        with torch.no_grad():
            tensorImageA, tensorDisparityA, tensorDepthA, zoom_from, zoom_to, dataset_ids = next(iter(self.data_loader))
            zoom_settings = {'objectFrom' : zoom_from, 'objectTo' : zoom_to}
            tensorImageA = tensorImageA.to(device, non_blocking=True)
            tensorDisparityA = tensorDisparityA.to(device, non_blocking=True)
            tensorDepthA = tensorDepthA.to(device, non_blocking=True)

            if not outputRenderC:
                tensorMasks, tensorShift, objectList = get_masks(tensorImageA, tensorDisparityA, tensorDepthA, zoom_settings, camera)

                tensorImageA = (tensorImageA + 1) / 2

                inpaintObject = self.moduleInpaint(tensorImage=tensorImageA * tensorMasks, 
                                                    tensorDisparity=tensorDisparityA * tensorMasks, 
                                                    tensorMasks=tensorMasks)

                inpaintImage = inpaintObject['tensorImage']
                inpaintDisparity = inpaintObject['tensorDisparity']

                return tensorImageA, inpaintImage, tensorDisparityA, inpaintDisparity, tensorMasks, zoom_settings

            else:
                tensorImageA = (tensorImageA + 1) / 2

                tensorImageA, tensorDisparityA = self.moduleInpaint.normalize_images_disp(tensorImageA, tensorDisparityA, not_normed=True)
                tensorContextA = self.moduleInpaint.moduleContext(torch.cat([ tensorImageA, tensorDisparityA ], 1))
                # tensorImageA, tensorDisparityA = self.moduleInpaint.normalize_images_disp(tensorImageA, tensorDisparityA, not_normed=False)

                tensorRenderB, tensorMaskB, tensorPointsA, tensorShift, objectList = get_masks(tensorImageA, tensorDisparityA, tensorDepthA, zoom_settings, camera,  
                                                                    AFromB=False, tensorContext=tensorContextA)

                tensorImageB, tensorDisparityB, tensorContextB = tensorRenderB[:,:3,:,:], tensorRenderB[:,3:4,:,:], tensorRenderB[:,4:,:,:]
                inpaintObjectB = self.moduleInpaint(tensorImage=tensorImageB,
                                                    tensorDisparity=tensorDisparityB, 
                                                    tensorMasks=tensorMaskB,
                                                    tensorContext=tensorContextB)
                
                
                
                # tensorRenderB, tensorMaskB, tensorPointsA, tensorShift, objectList = get_masks(tensorImageA, tensorDisparityA, tensorDepthA, zoom_settings, 
                #                                                     AFromB=False)
                # tensorImageB, tensorDisparityB = tensorRenderB[:,:3,:,:], tensorRenderB[:,3:4,:,:]

                
                # inpaintObjectB = self.moduleInpaint(tensorImage=tensorImageB,
                #                                     tensorDisparity=tensorDisparityB, 
                #                                     tensorMasks=tensorMaskB)

                inpaintImageB = inpaintObjectB['tensorImage']
                inpaintDisparityB = inpaintObjectB['tensorDisparity']
                # inpaintImageB[:,:,600:,950:] = torch.randn(1,3,156, 74).to(device)
                # inpaintDisparityB[:,:,:,950:] = 1

                inpaintDepthB = (camera['focal'] * camera['baseline']) / (inpaintDisparityB + 0.0000001)

                tensorImageA, tensorDisparityA = self.moduleInpaint.normalize_images_disp(tensorImageA, tensorDisparityA, not_normed=False)
                tensorRenderC, tensorMasksC = generate_new_view_from_inpaint(tensorPointsA, 
                                    tensorImageA, 
                                    tensorDisparityA, 
                                    tensorDepthA,
                                    inpaintImageB, 
                                    inpaintDisparityB, 
                                    inpaintDepthB, 
                                    tensorMaskB, 
                                    tensorShift,
                                    camera)

                
                tensorImageB, tensorDisparityB = self.moduleInpaint.normalize_images_disp(tensorImageB, tensorDisparityB, not_normed=False)
                # tensorImageC, tensorDisparityC = self.moduleInpaint.normalize_images_disp(tensorRenderC[:,:3,:,:], tensorRenderC[:,3:4,:,:], not_normed=False)
                tensorImageC, tensorDisparityC = tensorRenderC[:,:3,:,:], tensorRenderC[:,3:4,:,:]
                
                return (tensorImageA, 
                        tensorDisparityA, 
                        tensorImageB, 
                        tensorDisparityB, 
                        inpaintImageB, 
                        inpaintDisparityB, 
                        tensorImageC, 
                        tensorMasksC,
                        tensorDisparityC,
                        tensorMaskB, 
                        zoom_settings)


                


                                   