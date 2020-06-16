import getopt
import math
import os
import sys

import torch

from training.train_depth import TrainerDepth
from training.train_inpaint import TrainerInpaint
from utils.data_loader import Dataset

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
print('Number of threads used: ', torch.get_num_threads())

os.environ['CUDA_HOME'] = '/opt/cuda/cuda-10.1' # change this to your cuda installation path

#######################
# DATASET DEFINITIONs #
#######################

indoor_dataset_full = {'name':'diml-in-full',
                'path': '/scratch/s182169/DIML/full/indoor/',
                'params': {'focal': 512, 'baseline':74}}

outdoor_dataset_full = {'name':'diml-out-full',
                'path': '/scratch/s182169/DIML/full/outdoor/',
                'params': {'focal': 512, 'baseline':120}}

gta_dataset = {'name':'gta',
                'path': '/scratch/s182169/GTAV_1080/',
                'params': {'focal': 770, 'baseline':12}}

dataset_list = [indoor_dataset_full, outdoor_dataset_full, gta_dataset]


n_epochs = 100
training_mode = 'estimation'
mask_loss_mode = None
mask_loss_dataset = None
lr_estimation = 1e-4
lr_refinement = 1e-5
lr_inpaint = 1e-4
lr_discriminator = 5e-5

partial_conv = False

batch_size = 8
gamma_lr = 0.99999

save_name = '3dkbe'


strParameter = ['mask-loss=', 'mask-loss-dataset=', 'n-epochs=', 'lr-estimation=', 'lr-refinement=', 'lr-inpaint=', 
                'lr-discriminator=', 'save-name=', 'model-path=', 'batch-size=', 'gamma-lr=', 'partial-conv', 'training-mode=',
                'save-path=']

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', strParameter)[0]: 
    if strOption == '--training-mode' and strArgument != '' and strArgument in ['estimation', 'refinement', 'inpainting', 'inpainting_ref']: 
        training_mode = strArgument # mask loss mode can be 'none', 'same' or 'other'
    elif strOption == '--training-mode' and strArgument != '':
        print('Unknown training mode selected.')
        
    if strOption == '--mask-loss' and strArgument != '' and strArgument in ['none', 'same', 'other']: 
        mask_loss_mode = strArgument # mask loss mode can be 'none', 'same' or 'other'
    elif strOption == '--mask-loss' and strArgument != '':
        print('Unknown mask loss mode selected')
    
    if strOption == '--mask-loss-dataset' and strArgument != '': 
        mask_loss_dataset = strArgument # in case mask_loss_mode == 'other' cannot be None
    
    if strOption == '--lr-estimation' and strArgument != '': 
        lr_estimation = float(strArgument) # learning rate for estimation network

    if strOption == '--lr-refinement' and strArgument != '': 
        lr_refinement = float(strArgument) # learning rate for refinement network

    if strOption == '--lr-inpaint' and strArgument != '': 
        lr_inpaint = float(strArgument) # learning rate for inpainting network

    if strOption == '--lr-discriminator' and strArgument != '': 
        lr_discriminator = float(strArgument) # learning rate for discriminator network

    if strOption == '--model-path' and strArgument != '': 
        model_path = [strArgument] # model path for either continue training or refinement 

    if strOption == '--batch-size' and strArgument != '': 
        batch_size = int(strArgument) # batch size
    
    if strOption == '--partial-conv': 
        partial_conv = True # batch size
    
    if strOption == '--gamma-lr' and strArgument != '': 
        gamma_lr = float(strArgument) # batch size
    
    if strOption == '--gamma-lr' and strArgument != '': 
        save_name = strArgument # network save name, note that this will be combine with the type of network trained
    

if mask_loss_mode == 'other':
    assert mask_loss_dataset is not None, 'When computing the maskloss on a different dataset than the depth training please specify its path'
elif mask_loss_mode == 'none':
    mask_loss_mode = None

if training_mode is in ['refinement', 'inpainting_ref']:
    assert model_path is not None, 'Need path to pre-trained network for refinement training.'


if __name__ == '__main__':

    if training_mode == 'estimation':
        ## Train disparity network 
        train_depth = TrainerDepth(dataset_list, 
                                {'n_epochs':n_epochs, 
                                'gamma_lr':gamma_lr,
                                'batch_size':batch_size,
                                'model_to_train':'disparity',
                                'lr_estimation':lr_estimation,
                                'save_name':save_name,
                                'mask_loss': mask_loss_mode,
                                'mask_loss_path': mask_loss_dataset})
        train_depth.train()


    elif training_mode == 'refinement':
        ## Train depth refinement network
        train_depth = TrainerDepth(dataset_list, 
                                {'n_epochs':n_epochs, 
                                'gamma_lr':gamma_lr,
                                'batch_size':batch_size,
                                'model_to_train':'refine',
                                'lr_estimation':lr_estimation,
                                'lr_refine':lr_refinement,
                                'save_name':save_name},
                                 models_paths=model_path)
            
        train_depth.train()


    elif training_mode == 'inpainting':
        ## Train inpainting network
        model_to_train = 'inpainting'
        if partial_conv:
            model_to_train = 'partial inpainting'

        train_inpaint = TrainerInpaint(dataset_list, 
                                {'n_epochs':n_epochs, 
                                'gamma_lr':gamma_lr,
                                'batch_size':batch_size,
                                'model_to_train':model_to_train,
                                'lr_inpaint':lr_inpaint,
                                'adversarial':False,
                                'save_name':save_name})
        train_inpaint.train()


    elif training_mode == 'inpainting_ref':
        # Train inpainting with adversarial network
        model_to_train = 'inpainting'
        if partial_conv:
            model_to_train = 'partial inpainting'
            
        train_inpaint = TrainerInpaint(dataset_list, 
                                {'n_epochs':n_epochs, 
                                'gamma_lr':gamma_lr,
                                'batch_size':batch_size,
                                'model_to_train':'inpainting',
                                'lr_inpaint':lr_inpaint,
                                'lr_D':lr_discriminator,
                                'adversarial':True,
                                'save_name':save_name}, 
                                models_paths=model_path)
        train_inpaint.train()
