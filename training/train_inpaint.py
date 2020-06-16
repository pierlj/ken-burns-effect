import functools

import cv2
import dill
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from models.discriminator import *
from models.partial_inpainting import Inpaint as PartialInpaint
from models.pointcloud_inpainting import Inpaint
from utils.common import depth_to_points, generate_mask, spatial_filter
from utils.data_loader import Dataset
from utils.fid import FID
from utils.losses import *
from utils.utils import *


class TrainerInpaint():
    def __init__(self, dataset_paths, training_params, models_paths=None, logs_path='runs/train_0', continue_training=False):
        self.iter_nb = 0
        self.dataset_paths = dataset_paths
        self.training_params = training_params

        self.dataset = Dataset(dataset_paths, mode='inpainting')

        # Create training and validation set randomly
        dataset_length = len(self.dataset)
        train_set_length = int(0.99 * dataset_length)
        validation_set_length  = dataset_length - train_set_length

        self.training_set, self.validation_set = torch.utils.data.random_split(self.dataset, [train_set_length, validation_set_length])

        self.data_loader = torch.utils.data.DataLoader(self.training_set, 
                                                        batch_size=self.training_params['batch_size'], 
                                                        shuffle=True, 
                                                        pin_memory=True, 
                                                        num_workers=2)

        self.data_loader_validation = torch.utils.data.DataLoader(self.validation_set, 
                                                                batch_size=self.training_params['batch_size']*2, 
                                                                shuffle=True, 
                                                                pin_memory=True, 
                                                                num_workers=2)

        if self.training_params['model_to_train'] == 'inpainting':
            self.moduleInpaint = Inpaint().to(device).train()
        elif self.training_params['model_to_train'] == 'partial inpainting':
            self.moduleInpaint = PartialInpaint().to(device).train()

        weights_init(self.moduleInpaint, init_gain=0.01)
        
        self.optimizer_inpaint = torch.optim.Adam(self.moduleInpaint.parameters(), lr=self.training_params['lr_inpaint'])

        self.loss_inpaint = InpaintingLoss(kbe_only=False, perceptual=True)

        self.loss_weights = {'hole':6,
                             'valid':1,
                             'prc':0.05,
                             'tv': 0.1,
                             'style':120,
                             'grad':10,
                             'ord':0.0001,
                             'color':0,
                             'mask':0.0001,
                             'valid_depth':1,
                             'joint_edge':1}
        

        lambda_lr = lambda epoch: self.training_params['gamma_lr'] ** epoch
        self.scheduler_inpaint = torch.optim.lr_scheduler.LambdaLR(self.optimizer_inpaint, lr_lambda=lambda_lr)


        if models_paths is not None:
            models_list = [{'model':self.moduleInpaint,'type':'inpaint'}]
            load_models(models_list, models_paths)

        if self.training_params['adversarial']:
            ## Train with view B
            self.discriminator = MPDDiscriminator().to(device) # other type of discriminator can be used here

            ## Train with view C
            # self.discriminator = MultiScalePerceptualDiscriminator().to(device)
            
            spectral_norm_switch(self.discriminator, on=True)

            self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=self.training_params['lr_D'])
            self.schedulerD = torch.optim.lr_scheduler.LambdaLR(self.optimizerD, lr_lambda=lambda_lr)

            # discriminator balancing parameters
            self.balanceSteps = 5 # number of D steps per G step
            self.pretrainSteps = 1000 # number of pretraining steps for D
            self.stopG = 10000 # restart pretraining of D every stopG steps

        self.writer = CustomWriter(logs_path)

        
    
    def train(self):
        print('Starting training of inpaint net on datasets: ', functools.reduce(lambda s1, s2: s1 + '\n ' + s2['path'], self.dataset_paths, ""))
        self.moduleInpaint.train()

        if self.training_params['adversarial']:
            self.train_adversarial()
        else:
            self.train_inpaint()
         
        self.writer.add_hparams(self.training_params, {})
    
    def train_inpaint(self):
        camera = {'focal':1024/2.0, 'baseline':74.0}

        for epoch in range(self.training_params['n_epochs']):
            for idx, (tensorImage, tensorDisparity, tensorDepth, zoom_from, zoom_to, dataset_ids) in enumerate(tqdm(self.data_loader, desc='Epoch %d/%d'%(epoch +1 , self.training_params['n_epochs']))):
                zoom_settings = {'objectFrom' : zoom_from, 'objectTo' : zoom_to}
                
                if ((idx + 1) % 500) ==0:
                    save_model({'inpaint': {'model':self.moduleInpaint, 
                                'opt':self.optimizer_inpaint, 
                                'schedule': self.scheduler_inpaint,
                                'save_name': self.training_params['save_name']}}, self.iter_nb)

                    self.validation()

                tensorImage = tensorImage.to(device, non_blocking=True)
                tensorDisparity = tensorDisparity.to(device, non_blocking=True)
                tensorDepth = tensorDepth.to(device, non_blocking=True)

                tensorMasks, tensorShift, objectList = get_masks(tensorImage, tensorDisparity, tensorDepth, zoom_settings, camera)

                tensorImage = (tensorImage + 1) / 2
                inpaintObject = self.moduleInpaint(tensorImage=tensorImage * tensorMasks, 
                                                    tensorDisparity=tensorDisparity * tensorMasks, 
                                                    tensorMasks=tensorMasks)

                inpaintImage = inpaintObject['tensorImage']
                inpaintDisparity = inpaintObject['tensorDisparity']

                # compute the losses
                loss_dict = self.loss_inpaint(tensorImage * tensorMasks, tensorMasks, inpaintImage, tensorImage)
                loss_dict['ord'] = compute_loss_ord(inpaintDisparity, tensorDisparity, tensorMasks)
                loss_dict['grad'] = compute_loss_grad(inpaintDisparity, tensorDisparity, tensorMasks)
                
                inpaint_loss =  0
                for key, value in loss_dict.items():
                    inpaint_loss += self.loss_weights[key] * value
                    self.writer.add_scalar('Inpaint/' + key, value, self.iter_nb)

                self.writer.add_scalar('Inpaint/total' , inpaint_loss, self.iter_nb)

                # backward pass
                self.optimizer_inpaint.zero_grad()
                inpaint_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.moduleInpaint.parameters(), 1)
                self.optimizer_inpaint.step()
                self.scheduler_inpaint.step()

                # ## keep track of gradient magnitudes
                # for i, m in enumerate(self.moduleInpaint.modules()):
                #     if m.__class__.__name__ == 'Conv2d':
                #         g = m.weight.grad.view(-1)
                #         if g is not None:
                #             self.writer.add_scalar('Inpaint gradients/Conv {}'.format(i), torch.norm(g/g.size(0), p=1).item(), self.iter_nb)

                self.iter_nb += 1
    
    def train_adversarial(self):
        camera = {'focal':1024/2.0, 'baseline':74.0}

        for epoch in range(self.training_params['n_epochs']):
            for idx, (tensorImageA, tensorDisparityA, tensorDepthA, zoom_from, zoom_to, dataset_ids) in enumerate(tqdm(self.data_loader, desc='Epoch %d/%d'%(epoch +1 , self.training_params['n_epochs']))):
                zoom_settings = {'objectFrom' : zoom_from, 'objectTo' : zoom_to}

                if ((self.iter_nb + 1) % 500) == 0:
                    save_model({'inpaint': {'model':self.moduleInpaint, 
                                'opt':self.optimizer_inpaint, 
                                'schedule': self.scheduler_inpaint,
                                'save_name': self.training_params['save_name']}}, self.iter_nb)
                    
                    save_model({'discriminator': {'model':self.discriminator, 
                            'opt':self.optimizerD, 
                            'schedule': self.schedulerD,
                            'save_name': self.training_params['save_name']}}, self.iter_nb)

                    self.validation_adv()

                tensorImageA = tensorImageA.to(device, non_blocking=True)
                tensorDisparityA = tensorDisparityA.to(device, non_blocking=True)
                tensorDepthA = tensorDepthA.to(device, non_blocking=True)

                tensorImageA = (tensorImageA + 1) / 2

                # Forward pass, includes extracting context from view A and warp it to view B
                tensorImageA, tensorDisparityA = self.moduleInpaint.normalize_images_disp(tensorImageA, tensorDisparityA, not_normed=True)
                tensorContextA = self.moduleInpaint.moduleContext(torch.cat([ tensorImageA, tensorDisparityA ], 1))
                

                tensorRenderB, tensorMaskB, tensorPointsA, tensorShift, objectList = get_masks(tensorImageA, tensorDisparityA, tensorDepthA, zoom_settings, camera,
                                                                    AFromB=False, tensorContext=tensorContextA)            
                tensorImageB, tensorDisparityB, tensorContextB = tensorRenderB[:,:3,:,:], tensorRenderB[:,3:4,:,:], tensorRenderB[:,4:,:,:]

                inpaintObjectB = self.moduleInpaint(tensorImage=tensorImageB,
                                                    tensorDisparity=tensorDisparityB, 
                                                    tensorMasks=tensorMaskB,
                                                    tensorContext=tensorContextB)

                inpaintImageB = inpaintObjectB['tensorImage']
                inpaintDisparityB = inpaintObjectB['tensorDisparity']
                inpaintDepthB = (camera['focal'] * camera['baseline']) / (inpaintDisparityB + 0.0000001)

                tensorImageB, tensorDisparityB = self.moduleInpaint.normalize_images_disp(tensorImageB, tensorDisparityB, not_normed=False)
                tensorImageA, tensorDisparityA = self.moduleInpaint.normalize_images_disp(tensorImageA, tensorDisparityA, not_normed=False)

                ## Train with view C
                # tensorRenderC, tensorMasksC = generate_new_view_from_inpaint(tensorPointsA, 
                #                     tensorImageA, 
                #                     tensorDisparityA, 
                #                     tensorDepthA,
                #                     inpaintImageB, 
                #                     inpaintDisparityB, 
                #                     inpaintDepthB, 
                #                     tensorMaskB, 
                #                     tensorShift,
                #                     camera)
                # tensorImageC, _ = self.moduleInpaint.normalize_images_disp(tensorRenderC[:,:3,:,:], tensorRenderC[:,3:,:,:], not_normed=False)
                
                # compute step for moduleInpaint i.e. the generator
                if (self.iter_nb % self.stopG) > self.pretrainSteps and self.iter_nb % self.balanceSteps == 0:
                    lossGAdv = self.discriminator.adversarialLoss(inpaintImageB, inpaintDisparityB, isReal=True)
                    loss_dict = self.loss_inpaint.forward_adv(tensorImageB, tensorMaskB, inpaintImageB, inpaintDisparityB, tensorDisparityB)

                    ## Train with view C
                    # lossGAdv = self.discriminator.adversarialLoss(tensorImageC, isReal=True)
                    # loss_dict = self.loss_inpaint.forward_adv(tensorRenderC[:,:3,:,:], tensorMasksC, tensorImageC)

                    lossGValid = 0.0
                    for key, value in loss_dict.items():
                        lossGValid += self.loss_weights[key] * value
                        self.writer.add_scalar('Inpaint/' + key, value, self.iter_nb)

                    lossG = 10 * lossGValid + lossGAdv
                    self.optimizer_inpaint.zero_grad()
                    lossG.backward()
                    torch.nn.utils.clip_grad_norm_(self.moduleInpaint.parameters(), 1)
                    self.optimizer_inpaint.step()
                    self.writer.add_scalar('Inpaint/Adversarial G', lossG.cpu().item(), self.iter_nb)

                    for _ in range(self.balanceSteps): # done to apply same decay to G lr and D lr
                        self.scheduler_inpaint.step()

                    # keep track of gradient magnitude
                    # for i, m in enumerate(self.moduleInpaint.modules()):
                    #     if m.__class__.__name__ == 'Conv2d':
                    #         g = m.weight.grad
                    #         # print(g)
                    #         if g is not None:
                    #             self.writer.add_scalar('G_gradient/Conv {}'.format(i), torch.norm(g/g.size(0), p=1).item(), self.iter_nb)


                # compute step for discriminator
                lossFakeD = self.discriminator.adversarialLoss(inpaintImageB.detach(), inpaintDisparityB.detach(), isReal=False)

                ## Train with view C
                # lossFakeD = self.discriminator.adversarialLoss(tensorImageC.detach(), isReal=False)
                

                lossRealD = self.discriminator.adversarialLoss(tensorImageA, tensorDisparityA, isReal=True)
                ## Train with view C
                # lossRealD = self.discriminator.adversarialLoss(tensorImageA, isReal=True)

                lossD = 0.5 * (lossFakeD + lossRealD)

                self.optimizerD.zero_grad()
                lossD.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1)
                self.optimizerD.step()

                self.writer.add_scalar('Inpaint/Adversarial D', lossD.cpu().item(), self.iter_nb)
                
                ## Change LR
                self.schedulerD.step()

                self.iter_nb += 1
                

    def validation(self):
        # Compute different metrics on the validation set in the supervised setting
        # PSNR image, PSNR disp, SSIM image, SSIM disp, FID
        self.moduleInpaint.eval()

        measures = []

        metrics = {}
        metrics_list = ['PSNR Image', 'PSNR Disparity', 'SSIM Image ', 'SSIM Disparity']
        camera = {'focal':1024/2.0, 'baseline':74.0}


        for idx, (tensorImage, tensorDisparity, tensorDepth, zoom_from, zoom_to, dataset_ids) in enumerate(tqdm(self.data_loader_validation, desc='Validation')):
            with torch.no_grad():
                zoom_settings = {'objectFrom' : zoom_from, 'objectTo' : zoom_to}

                tensorImage = tensorImage.to(device, non_blocking=True)
                tensorDisparity = tensorDisparity.to(device, non_blocking=True)
                tensorDepth = tensorDepth.to(device, non_blocking=True)

                tensorMasks, tensorShift, objectList = get_masks(tensorImage, tensorDisparity, tensorDepth, zoom_settings, camera)
                tensorImage = (tensorImage + 1) / 2

                inpaintObject = self.moduleInpaint(tensorImage=tensorImage * tensorMasks, 
                                                    tensorDisparity=tensorDisparity * tensorMasks, 
                                                    tensorMasks=tensorMasks)

                inpaintImage = inpaintObject['tensorImage']
                inpaintDisparity = inpaintObject['tensorDisparity']

                batch_metrics = compute_inpaint_metrics(inpaintImage, inpaintDisparity, tensorImage, tensorDisparity, tensorMasks)
                inpaintImage = torch.clamp(inpaintImage, 0.0, 1.0)
                measures.append(np.array(batch_metrics))
        
        measures = np.array(measures).mean(axis=0)

        for i, name in enumerate(metrics_list):
            metrics[name] = measures[i]
            self.writer.add_scalar('Validation inpaint/' + name, measures[i], self.iter_nb)
        
        self.moduleInpaint.train()

    def validation_adv(self):
        # Compute different metrics on the validation set in the unsupervised setting: only FID
        self.moduleInpaint.eval()
        
        camera = {'focal':1024/2.0, 'baseline':74.0}

        self.fidNetwork = FID()

        inception_activations_real = np.zeros((len(self.validation_set), 2048), dtype=np.float32)
        inception_activations_inpaint = np.zeros((len(self.validation_set), 2048), dtype=np.float32)

        for idx, (tensorImageA, tensorDisparityA, tensorDepthA, zoom_from, zoom_to, dataset_ids) in enumerate(tqdm(self.data_loader_validation, desc='Validation')):
            with torch.no_grad():
                zoom_settings = {'objectFrom' : zoom_from, 'objectTo' : zoom_to}

                batch_size = tensorImageA.shape[0]
                start_idx = batch_size * idx
                end_idx = batch_size * (idx + 1)

                tensorImageA = tensorImageA.to(device, non_blocking=True)
                tensorDisparityA = tensorDisparityA.to(device, non_blocking=True)
                tensorDepthA = tensorDepthA.to(device, non_blocking=True)

                tensorImageA = (tensorImageA + 1) / 2
                tensorImageA, tensorDisparityA = self.moduleInpaint.normalize_images_disp(tensorImageA, tensorDisparityA, not_normed=True)
                tensorContextA = self.moduleInpaint.moduleContext(torch.cat([ tensorImageA, tensorDisparityA ], 1))
                

                tensorRenderB, tensorMaskB, tensorPointsA, tensorShift, objectList = get_masks(tensorImageA, tensorDisparityA, tensorDepthA, zoom_settings, camera, 
                                                                    AFromB=False, tensorContext=tensorContextA)              
                tensorImageB, tensorDisparityB, tensorContextB = tensorRenderB[:,:3,:,:], tensorRenderB[:,3:4,:,:], tensorRenderB[:,4:,:,:]
                inpaintObjectB = self.moduleInpaint(tensorImage=tensorImageB,
                                                    tensorDisparity=tensorDisparityB, 
                                                    tensorMasks=tensorMaskB,
                                                    tensorContext=tensorContextB)


                inpaintImageB = inpaintObjectB['tensorImage']
                inpaintDisparityB = inpaintObjectB['tensorDisparity']
                inpaintDepthB = (camera['focal'] * camera['baseline']) / (inpaintDisparityB + 0.0000001)

                tensorImageA, tensorDisparityA = self.moduleInpaint.normalize_images_disp(tensorImageA, tensorDisparityA, not_normed=False)
                tensorImageB, tensorDisparityB = self.moduleInpaint.normalize_images_disp(tensorImageB, tensorDisparityB, not_normed=False)
                
                # when training with halfaway view C
                # tensorRenderC, _ = generate_new_view_from_inpaint(tensorPointsA, 
                #                     tensorImageA, 
                #                     tensorDisparityA, 
                #                     tensorDepthA,
                #                     inpaintImageB, 
                #                     inpaintDisparityB, 
                #                     inpaintDepthB, 
                #                     tensorMaskB, 
                #                     tensorShift, 
                #                     camera)

                # tensorImageA, tensorDisparityA = self.moduleInpaint.normalize_images_disp(tensorImageA, tensorDisparityA, not_normed=False)
                # tensorImageC, tensorDisparityC = self.moduleInpaint.normalize_images_disp(tensorRenderC[:,:3,:,:], tensorRenderC[:,3:4,:,:], not_normed=False)
                tensorImageA = torch.clamp(tensorImageA, 0.0, 1.0)
                tensorImageB = torch.clamp(tensorImageB, 0.0, 1.0)
                inpaintImageB = torch.clamp(inpaintImageB, 0.0, 1.0)

                # when training with halfaway view C
                # tensorImageC = torch.clamp(tensorImageC, 0.0, 1.0)

                images_processed = self.fidNetwork.preprocess_images(tensorImageA.cpu().permute(0,2,3,1).numpy(), False).to(device)
                
                images_inpaint_processed = self.fidNetwork.preprocess_images(inpaintImageB.cpu().permute(0,2,3,1).numpy(), False).to(device)
                # when training with halfaway view C
                # images_inpaint_processed = self.fidNetwork.preprocess_images(tensorImageC.cpu().permute(0,2,3,1).numpy(), False).to(device)

                activations_real = self.fidNetwork.inception_network(images_processed).detach().cpu().numpy()
                assert activations_real.shape == (images_processed.shape[0], 2048)
                activations_inpaint =  self.fidNetwork.inception_network(images_inpaint_processed).detach().cpu().numpy()
                assert activations_inpaint.shape == (images_inpaint_processed.shape[0], 2048)

                inception_activations_real[start_idx:end_idx, :] = activations_real
                inception_activations_inpaint[start_idx:end_idx, :] = activations_inpaint
        
        mu_real, sigma_real = self.fidNetwork.calculate_activation_statistics(inception_activations_real)
        mu_inpaint, sigma_inpaint =  self.fidNetwork.calculate_activation_statistics(inception_activations_inpaint)

        fid = self.fidNetwork.calculate_frechet_distance(mu_real, sigma_real, mu_inpaint, sigma_inpaint)

        self.writer.add_scalar('Validation inpaint/FID', fid, self.iter_nb)
        
        self.moduleInpaint.train()
