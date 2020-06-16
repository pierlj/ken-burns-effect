import torch
import math
import os
import sys
import getopt

from utils.data_loader import Dataset
from utils.pipeline import Pipeline
from training.eval_depth import DepthEval
from training.train_depth import TrainerDepth

from training.train_inpaint import TrainerInpaint



# torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
print('Number of threads used: ', torch.get_num_threads())


os.environ['CUDA_HOME'] = '/opt/cuda/cuda-10.1'


indoor_dataset = {'name':'diml-in',
                'path': '/scratch/s182169/DIML/indoor/train/prepared/',
                'params': {'focal': 512, 'baseline':74}} #74 

indoor_dataset_full = {'name':'diml-in-full',
                'path': '/scratch/s182169/DIML/full/indoor/',
                'params': {'focal': 512, 'baseline':74}}

indoor_test_dataset = {'name':'diml-in',
                'path': '/scratch/s182169/DIML/indoor/test/prepared/',
                'params': {'focal': 512, 'baseline':74}} #74 

outdoor_dataset = {'name':'diml-out',
                'path': '/scratch/s182169/DIML/outdoor/train/prepared/',
                'params': {'focal': 512, 'baseline':120}}

outdoor_dataset_full = {'name':'diml-out-full',
                'path': '/scratch/s182169/DIML/full/outdoor/',
                'params': {'focal': 512, 'baseline':120}}

mega_dataset = {'name':'mega',
                'path': '/scratch/s182169/MegaDepth/train/',
                'params': {'focal': 50, 'baseline':1}}

mega_dataset_test = {'name':'mega',
                'path': '/scratch/s182169/MegaDepth/test/',
                'params': {'focal': 50, 'baseline':1}}

gta_dataset = {'name':'gta',
                'path': '/scratch/s182169/GTAV_1080/',
                'params': {'focal': 770, 'baseline':12}}

# indoor_dataset = {'path': '/scratch/s182169/DIML/indoor/test/prepared/',
#                 'params': {'focal': 512, 'baseline':40}}

# outdoor_dataset = {'path': '/scratch/s182169/DIML/outdoor/test/prepared/',
#                 'params': {'focal': 512, 'baseline':40}}

# nyuv2_dataset = {'name':'nyu','path': '/scratch/s182169/NYUv2/test/', 'params': {'focal': 320, 'baseline':40}}

## Train disparity network 

train_depth = TrainerDepth([indoor_dataset, outdoor_dataset, indoor_dataset_full, outdoor_dataset_full, gta_dataset], 
                        {'n_epochs':200, 
                        'gamma_lr':0.999998,
                        'batch_size':16,
                        'model_to_train':'disparity',
                        'lr_estimation':1e-4,
                        'lr_refine':1e-5,
                        'save_name':'in-out-full-1M',
                        'mask_loss': None})
train_depth.train()



# Train refine network

# train_depth = TrainerDepth([indoor_dataset, outdoor_dataset, indoor_dataset_full, outdoor_dataset_full], 
#                         {'n_epochs':10, 
#                         'gamma_lr':0.99999,
#                         'batch_size':2,
#                         'model_to_train':'refine',
#                         'lr_estimation':1e-4,
#                         'lr_refine':1e-4,
#                         'save_name':'in-out-full-2',
#                         'mask_loss': None},
#                          models_paths=['./models/trained/disparity-estimation-in-out-full.tar'])
    
# train_depth.train()


## Train inpainting network

# train_inpaint = TrainerInpaint([indoor_dataset, outdoor_dataset, indoor_dataset_full, outdoor_dataset_full], 
#                         {'n_epochs':10, 
#                         'gamma_lr':0.99999,
#                         'batch_size':2,
#                         'model_to_train':'partial inpainting',
#                         'lr_inpaint':1e-4,
#                         'adversarial':False,
#                         'save_name':'in-out-partial-or'})

# Train inpainting with adversarial network

# train_inpaint = TrainerInpaint([indoor_dataset, outdoor_dataset, indoor_dataset_full, outdoor_dataset_full], 
#                         {'n_epochs':300, 
#                         'gamma_lr':0.99999,
#                         'batch_size':2,
#                         'model_to_train':'inpainting',
#                         'lr_inpaint':1e-5,
#                         'lr_D':5e-5,
#                         'adversarial':True,
#                         'save_name':'in-out-adversarial-pretrained'}, models_paths=['./models/Pretrained_models/pointcloud-inpainting.pytorch'])
#                         # 'save_name':'in-out-adversarial-partial'}, models_paths=['./models/trained/inpaint-in-out-partial-fully.tar'])
#                         # 'save_name':'in-out-adversarial-maskfull'}, models_paths=['./models/trained/inpaint-in-out-regular.tar'])
                        

# train_inpaint.train()

