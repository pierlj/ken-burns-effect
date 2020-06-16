import functools
import sys

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from models.disparity_estimation import Disparity, Semantics
from models.disparity_refinement import Refine
from utils.data_loader import Dataset
from utils.losses import *
from utils.utils import *


class TrainerDepth():
    def __init__(self, dataset_paths, training_params, models_paths=None, logs_path='runs/train_0', continue_training=False):
        self.iter_nb = 0
        self.dataset_paths = dataset_paths
        self.training_params = training_params

        self.dataset = Dataset(dataset_paths, imagenet_path=self.training_params['mask_loss_path'])

        torch.manual_seed(111)
        np.random.seed(42)

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
                                                                batch_size=self.training_params['batch_size'], 
                                                                shuffle=True, 
                                                                pin_memory=True, 
                                                                num_workers=2)


        self.moduleSemantics = Semantics().to(device).eval()
        self.moduleDisparity = Disparity().to(device).eval()

        weights_init(self.moduleDisparity)

        self.moduleMaskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()
        
        self.optimizer_disparity = torch.optim.Adam(self.moduleDisparity.parameters(), lr=self.training_params['lr_estimation'])

        lambda_lr = lambda epoch: self.training_params['gamma_lr'] ** epoch
        self.scheduler_disparity = torch.optim.lr_scheduler.LambdaLR(self.optimizer_disparity, lr_lambda=lambda_lr)

        if self.training_params['model_to_train'] == 'refine' or self.training_params['model_to_train'] == 'both':
            self.moduleRefine = Refine().to(device).eval()
            weights_init(self.moduleRefine)
            self.optimizer_refine = torch.optim.Adam(self.moduleRefine.parameters(), lr=self.training_params['lr_refine'])
            self.scheduler_refine = torch.optim.lr_scheduler.LambdaLR(self.optimizer_refine, lr_lambda=lambda_lr)

        if models_paths is not None:
            print('Loading model state from '+ str(models_paths))
            if self.training_params['model_to_train'] == 'refine' or self.training_params['model_to_train'] == 'both':
                models_list = [{'model':self.moduleDisparity,
                                'type':'disparity',
                                'opt':self.optimizer_disparity,
                                'schedule':self.scheduler_disparity},
                            {'model':self.moduleRefine,
                                'type':'refine',
                                'opt':self.optimizer_refine,
                                'schedule':self.scheduler_refine}]
            else:
                models_list = [{'model':self.moduleDisparity,
                                'type':'disparity',
                                'opt':self.optimizer_disparity,
                                'schedule':self.scheduler_disparity}]
            self.iter_nb = load_models(models_list, models_paths, continue_training=continue_training)


        # use tensorboard to keep track of the runs
        self.writer = CustomWriter(logs_path)

        
    
    def train(self):

        print('Starting training of estimation net on datasets: ', functools.reduce(lambda s1, s2: s1 + '\n ' + s2['path'], self.dataset_paths, ""))
        

        if self.training_params['model_to_train'] == 'disparity':
            print('Training disparity estimation network.')
            self.moduleDisparity.train()
        elif self.training_params['model_to_train'] == 'refine':
            print('Training disparity refinement network.')
            self.moduleRefine.train()
        elif self.training_params['model_to_train'] == 'both':
            print('Training disparity networks.')
            self.moduleDisparity.train()
            self.moduleRefine.train()
        
        
        if self.training_params['model_to_train'] == 'refine' or self.training_params['model_to_train'] == 'both':
            self.train_refine()
        
        elif self.training_params['model_to_train'] == 'disparity':
            self.train_estimation()
                
        self.writer.add_hparams(self.training_params, {})

    
    def train_estimation(self):
        for epoch in range(self.training_params['n_epochs']):
            for idx, (tensorImage, GTdisparities, sparseMask, imageNetTensor, dataset_ids) in enumerate(tqdm(self.data_loader, desc='Epoch %d/%d'%(epoch +1 , self.training_params['n_epochs']))):
                if ((idx + 1) % 500) ==0:
                    save_model({'disparity': {'model':self.moduleDisparity, 
                                'opt':self.optimizer_disparity, 
                                'schedule': self.scheduler_disparity,
                                'save_name': self.training_params['save_name']}}, self.iter_nb)
                    self.validation()
    
                tensorImage = tensorImage.to(device, non_blocking=True)
                GTdisparities = GTdisparities.to(device, non_blocking=True)
                sparseMask = sparseMask.to(device, non_blocking=True)
                imageNetTensor = imageNetTensor.to(device, non_blocking=True)

                with torch.no_grad():
                    semantic_tensor = self.moduleSemantics(tensorImage)

                # forward pass
                tensorDisparity = self.moduleDisparity(tensorImage, semantic_tensor)  # depth estimation
                tensorDisparity = F.threshold(tensorDisparity, threshold=0.0, value=0.0)
                
                # reconstruction loss computation
                estimation_loss_ord = compute_loss_ord(tensorDisparity, GTdisparities, sparseMask, mode='logrmse')
                estimation_loss_grad = compute_loss_grad(tensorDisparity, GTdisparities, sparseMask)

                # loss weights computation
                beta = 0.015
                gamma_ord = 0.03 * (1+ 2 * np.exp( - beta * self.iter_nb)) # for scale-invariant Loss 
                # gamma_ord = 0.001 * (1+ 200 * np.exp( - beta * self.iter_nb)) # for L1 loss
                gamma_grad = 1 - np.exp( - beta * self.iter_nb)
                gamma_mask = 0.0001 *(1 - np.exp( - beta * self.iter_nb))

                if self.training_params['mask_loss'] == 'same':
                    # when mask_loss is 'same' masks are computed on the same images 
                    with torch.no_grad():
                        objectPredictions = self.moduleMaskrcnn(tensorImage)

                    masks_tensor_list = list(map(lambda object_pred: resize_image(object_pred['masks'], max_size=256), objectPredictions))
                    estimation_masked_loss = 0
                    for i, masks_tensor in enumerate(masks_tensor_list):
                        if masks_tensor is not None:
                            estimation_masked_loss += compute_masked_grad_loss(tensorDisparity[i].view(1,*tensorDisparity[i].shape),
                                                                                    masks_tensor, [1], 0.5)
                 
                    loss_depth = gamma_ord * estimation_loss_ord + gamma_grad * estimation_loss_grad + gamma_mask * estimation_masked_loss

                else: # No mask loss in this case
                    loss_depth = gamma_ord * estimation_loss_ord + gamma_grad * estimation_loss_grad
                    
                # compute gradients and update net
                self.optimizer_disparity.zero_grad()
                loss_depth.backward()
                torch.nn.utils.clip_grad_norm_(self.moduleDisparity.parameters(), 1)
                self.optimizer_disparity.step()
                self.scheduler_disparity.step()

                # keep track of loss values
                self.writer.add_scalar('Estimation/Loss ord', estimation_loss_ord, self.iter_nb)
                self.writer.add_scalar('Estimation/Loss grad', estimation_loss_grad, self.iter_nb)
                self.writer.add_scalar('Estimation/Loss depth', loss_depth, self.iter_nb)
                
                if self.training_params['mask_loss'] == 'same':
                    self.writer.add_scalar('Estimation/Loss mask', estimation_masked_loss, self.iter_nb)
                elif self.training_params['mask_loss'] == 'other':
                    self.step_imagenet(imageNetTensor) # when mask loss is computed on another dataset
                else:
                    self.writer.add_scalar('Estimation/Loss mask', 0, self.iter_nb)

                # keep track of gradient magnitudes
                # for i, m in enumerate(self.moduleDisparity.modules()):
                #     if m.__class__.__name__ == 'Conv2d':
                #         g = m.weight.grad
                #         # print(g)
                #         if g is not None:
                #             self.writer.add_scalar('Estimation gradients/Conv {}'.format(i), torch.norm(g/g.size(0), p=1).item(), self.iter_nb)
                
                self.iter_nb += 1
            
            self.validation()
             
    
    def train_refine(self):

        self.dataset.mode = 'refine'
        self.data_loader = self.dataset.get_dataloader(batch_size=2)

        for epoch in range(self.training_params['n_epochs']):
            for idx, (tensorImage, GTdisparities, masks, imageNetTensor, dataset_ids) in enumerate(tqdm(self.data_loader, desc='Epoch %d/%d'%(epoch +1 , self.training_params['n_epochs']))):
                if ((idx + 1) %500) == 0:
                    save_model({'refine': {'model':self.moduleRefine, 
                                'opt':self.optimizer_refine, 
                                'schedule': self.scheduler_refine,
                                'save_name': self.training_params['save_name']}}, self.iter_nb)
                    self.validation(refine_training=True)

                tensorImage = tensorImage.to(device, non_blocking=True)
                GTdisparities = GTdisparities.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                # first estimate depth with estimation net
                with torch.no_grad():
                    tensorResized = resize_image(tensorImage, max_size=512)
                    tensorDisparity = self.moduleDisparity(tensorResized, self.moduleSemantics(tensorResized))  # depth estimation
                    tensorResized = None

                # forward pass with refinement net
                tensorDisparity = self.moduleRefine(tensorImage, tensorDisparity)
                
                # compute losses
                refine_loss_ord = compute_loss_ord(tensorDisparity, GTdisparities, masks)
                refine_loss_grad = compute_loss_grad(tensorDisparity, GTdisparities, masks)

                loss_depth =  0.0001 * refine_loss_ord + refine_loss_grad              
                
                if self.training_params['model_to_train'] =='both':
                    self.optimizer_disparity.zero_grad()

                # backward pass
                self.optimizer_refine.zero_grad()
                loss_depth.backward()
                torch.nn.utils.clip_grad_norm_(self.moduleRefine.parameters(), 1)
                self.optimizer_refine.step()
                self.scheduler_refine.step()

                if self.training_params['model_to_train'] =='both':
                    self.optimizer_disparity.step()

                ## keep track of loss on tensorboard
                self.writer.add_scalar('Refine/Loss ord', refine_loss_ord, self.iter_nb)
                self.writer.add_scalar('Refine/Loss grad', refine_loss_grad, self.iter_nb)
                self.writer.add_scalar('Refine/Loss depth', loss_depth, self.iter_nb)
                
                ## keep track of gradient magnitudes
                # for i, m in enumerate(self.moduleRefine.modules()):
                #     if m.__class__.__name__ == 'Conv2d':
                #         g = m.weight.grad.view(-1)
                #         if g is not None:
                #             self.writer.add_scalar('Refine gradients/Conv {}'.format(i), torch.norm(g/g.size(0), p=1).item(), self.iter_nb)
                
                self.iter_nb += 1
    
    
    def step_imagenet(self, tensorImage):
        with torch.no_grad():
            semantic_tensor = self.moduleSemantics(tensorImage)

        tensorDisparity = self.moduleDisparity(tensorImage, semantic_tensor)  # depth estimation

        # compute segmentation masks on batch
        with torch.no_grad():
            objectPredictions = self.moduleMaskrcnn(tensorImage)

        masks_tensor_list = list(map(lambda object_pred: resize_image(object_pred['masks'], max_size=256), objectPredictions))        
        
        # compute mask loss
        estimation_masked_loss = 0
        for i, masks_tensor in enumerate(masks_tensor_list):
            if masks_tensor is not None:
                estimation_masked_loss += 0.0001 * compute_masked_grad_loss(tensorDisparity[i].view(1,*tensorDisparity[i].shape),
                                                                        resize_image(masks_tensor.view(-1,1,256,256), max_size=128), [1],  1)
        
        if estimation_masked_loss != 0:
            # backward pass for mask loss only
            self.optimizer_disparity.zero_grad()
            estimation_masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.moduleDisparity.parameters(), 0.1)
            self.optimizer_disparity.step()
            self.scheduler_disparity.step()

            self.writer.add_scalar('Estimation/Loss mask', estimation_masked_loss, self.iter_nb)

  
    def validation(self, refine_training=False):
        # compute metrics on the validation set
        self.moduleDisparity.eval()
        if refine_training:
            self.moduleRefine.eval()

        measures = []

        metrics = {}
        metrics_list = ['Abs rel', 'Sq rel', 'RMSE', 'log RMSE', 's1', 's2', 's3']
        MSELoss = nn.MSELoss()

        with torch.no_grad():
            for idx, (tensorImage, disparities, masks, imageNetTensor, dataset_ids) in enumerate(tqdm(self.data_loader_validation, desc='Validation')):
                tensorImage = tensorImage.to(device, non_blocking=True)
                disparities = disparities.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                tensorResized = resize_image(tensorImage)
                tensorDisparity = self.moduleDisparity(tensorResized, self.moduleSemantics(tensorResized))  # depth estimation

                if refine_training:
                    tensorDisparity = self.moduleRefine(tensorImage, tensorDisparity) # increase resolution
                else:
                    disparities = resize_image(disparities, max_size=256)
                    masks = resize_image(masks, max_size=256)
                
                tensorDisparity = F.threshold(tensorDisparity, threshold=0.0, value=0.0)

                masks = masks.clamp(0,1)
                measures.append(np.array(compute_metrics(tensorDisparity, disparities, masks)))
    
        
        measures = np.array(measures).mean(axis=0)

        for i, name in enumerate(metrics_list):
            metrics[name] = measures[i]
            self.writer.add_scalar('Validation/' + name, measures[i], self.iter_nb)
        
        if refine_training:
            self.moduleRefine.train()
        else:
            self.moduleDisparity.train()
