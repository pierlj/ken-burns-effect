import os

import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import remove_spectral_norm, spectral_norm
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from utils.common import (depth_to_points, fill_disocclusion, generate_mask,
                          process_shift, render_pointcloud, spatial_filter)

cuda = torch.cuda.is_available()
device = "cuda:0" if cuda else "cpu"

# Plotting function used to display image in a grid manner
def plot_all(images, n_img = 1, n_rows = 3, figsize=(16,8)):
    grid_list = []

    for img in images:
        if img.size()[0]==1:
            img = img.expand(3, *img.size()[1:])
        grid_list.append(img)
    grid_img = torchvision.utils.make_grid(grid_list, padding = 10, nrow = n_rows, normalize=False)
    grid_img = grid_img / 2 + 0.5     # unnormalize
    npimg = grid_img.numpy()
    plt.figure(figsize = figsize)
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def normalize_depth(depth):
    depth = (depth + 1) / 2
    depth = depth/depth.max()
    return depth * 2 - 1

def normalize_torch_tensor(tensor):
    return (tensor + 1) / 2


class CustomWriter(SummaryWriter):
    def __init__(self, path_name='runs/eval', foldername=None):
        while os.path.isdir(path_name):
            try:
                run_id, run_name = path_name[::-1].split('_', 1)
                run_name = run_name[::-1]
                run_id = int(run_id[::-1][:-1])
                path_name = run_name + '_' + str(run_id+1) + '/'

            except:
                path_name = path_name + "_1/"
        if foldername is not None:
            path_name = os.path.join(path_name, foldername)
        print('Logs will be saved in folder:', path_name)
        super(CustomWriter, self).__init__(path_name)

def resize_image(tensorImage, max_size=512):
    if tensorImage.size(0) == 0:
        return None
    intWidth = tensorImage.size(3)
    intHeight = tensorImage.size(2)

    dblRatio = float(intWidth) / float(intHeight)

    intWidth = min(int(max_size * dblRatio), max_size)
    intHeight = min(int(max_size / dblRatio), max_size)
    
    tensorImage = F.interpolate(input=tensorImage, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    return tensorImage

# Create kernels used for gradient computation
def get_kernels(h):
    kernel_elements = [-1] + [0 for _ in range(h-1)] + [1]
    kernel_elements_div = [1] + [0 for _ in range(h-1)] + [1]
    weight_y = torch.Tensor(kernel_elements).view(1,-1)
    weight_x = weight_y.T
    
    weight_y_norm = torch.Tensor(kernel_elements_div).view(1,-1)
    weight_x_norm = weight_y_norm.T
    
    return weight_x.to(device), weight_x_norm.to(device), weight_y.to(device), weight_y_norm.to(device)

def derivative_scale(imageTensor, h, norm=True):
    kernel_x, kernel_x_norm, kernel_y, kernel_y_norm = get_kernels(h)
    
    diff_x = torch.nn.functional.conv2d(imageTensor, kernel_x.view(1,1,-1,1))
    diff_y = torch.nn.functional.conv2d(imageTensor, kernel_y.view(1,1,1,-1))

    if norm:
        norm_x = torch.nn.functional.conv2d(torch.abs(imageTensor), kernel_x_norm.view(1,1,-1,1))
        norm_y = torch.nn.functional.conv2d(torch.abs(imageTensor), kernel_y_norm.view(1,1,1,-1))
        diff_x = diff_x/(norm_x + 1e-7)
        diff_y = diff_y/(norm_y + 1e-7)
    
    return torch.nn.functional.pad(diff_x, (0, 0, h, 0)), torch.nn.functional.pad(diff_y, (h, 0, 0, 0))

def weights_init(model, init_type='xavier', init_gain=1.4):
        def initialization(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and classname.find('Conv') != -1:
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain = init_gain)
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain = init_gain)
                elif init_type == 'he':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if init_type != 'None':
            model.apply(initialization)

# Computation of depth estimation metrics 
def compute_metrics(depth, depth_gt, masks):
    depth = depth * masks + 1e-7
    depth_gt = depth_gt * masks + 1e-7
    
    thresh = torch.max((depth_gt / depth), (depth / depth_gt))
    a1 = (thresh < 1.25     ).type(torch.FloatTensor).mean()
    a2 = (thresh < 1.25 ** 2).type(torch.FloatTensor).mean()
    a3 = (thresh < 1.25 ** 3).type(torch.FloatTensor).mean()

    rmse = (depth_gt - depth) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(depth_gt) - torch.log10(depth)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(depth_gt - depth) / depth_gt)

    sq_rel = torch.mean(((depth_gt - depth) ** 2) / depth_gt)

    return abs_rel.item(), sq_rel.item(), rmse.item(), rmse_log.item(), a1.item(), a2.item(), a3.item()


# Computation of inpainting metrics
def compute_inpaint_metrics(imageInpaint, disparityInpaint, imageGT, disparityGT, masks):

    def psnr(im1, im2, disp=False):
        mse = torch.mean((im1 - im2)**2)
        d = 512 if disp else 1
        psnr =  20 * torch.log10(d**2 / torch.sqrt(mse))
        return psnr

    psnrImg = psnr(imageInpaint, imageGT).cpu().item()
    psnrDisp = psnr(disparityInpaint, disparityGT, disp=True).cpu().item()

    ssim = kornia.losses.SSIM(window_size=11, reduction='mean')

    ssimImg = ssim(imageInpaint, imageGT).cpu().item()
    ssimDisp = ssim(disparityInpaint, disparityGT).cpu().item()

    return psnrImg, psnrDisp, ssimImg, ssimDisp


def class_to_masks(pred):
    batch_masks = []
    for img in pred:
        masks = []
        for c in img.unique():
            if c != 0:
                masks.append((img == c).type(torch.FloatTensor).unsqueeze(0))
        if len(masks)>0:
            batch_masks.append(torch.cat(masks, dim=0).to(device))
        else:
            batch_masks.append(torch.zeros(1,256, 256).to(device))
    return batch_masks


def sum_item(tensor):
    return torch.sum(tensor, dim=[d for d in range(1, len(tensor.shape))])

def spectral_norm_switch(model, on=True):
    if not model.spectral_norm and on:
        model.spectral_norm = True
        for m in model.modules():
            if hasattr(m, 'weight') and 'Conv' in m.__class__.__name__:
                m = spectral_norm(m)
    elif model.spectral_norm and not on:
        model.spectral_norm = False
        for m in model.modules():
            if hasattr(m, 'weight') and 'Conv' in m.__class__.__name__:
                m = remove_spectral_norm(m)



def save_model(models_dict, nb_iter, path='models/trained'):
    
    for model_type, model in models_dict.items():
        torch.save({
            'nb_iter': nb_iter,
            'model_state_dict': model['model'].state_dict(),
            'optimizer_'+ model_type +'_state_dict': model['opt'].state_dict(),
            'scheduler_'+ model_type +'_state_dict': model['schedule'].state_dict()}, 
            path + '/' + model_type + '-'+ model['save_name'] + '.tar')
    

    
def load_models(models_list, models_paths, continue_training=False):
    iter_nb = 0
    print('Loading models parameters...')
    for idx, model in enumerate(models_paths):
        try:
            checkpoint_disparity = torch.load(model)
            models_list[idx]['model'].load_state_dict(checkpoint_disparity['model_state_dict'])
            if continue_training:
                models_list[idx]['opt'].load_state_dict(checkpoint_disparity['optimizer_'+ models_list[idx]['type'] +'_state_dict'])
                models_list[idx]['schedule'].load_state_dict(checkpoint_disparity['scheduler_'+ models_list[idx]['type'] +'_state_dict'])
                iter_nb = checkpoint_disparity['nb_iter']
            print('Model ' + models_list[idx]['type'] + ' loaded succesfully.')
        except:
            models_list[idx]['model'].load_state_dict(torch.load(model))
            print('Pre-trained model ' + models_list[idx]['type'] + ' loaded succesfully.')
    return iter_nb


# Compute camera motion from zoom settings and a specific RGB-D image
def get_tensor_shift(objectCommon):
    dblStep = 1
    dblFrom = 1.0 - dblStep
    dblTo = 1.0 - dblFrom

    zoomSettings = objectCommon['zoomSettings']

    ## Current shift with respect to center of the image (hence -width/2)
    dblShiftU = ((dblFrom * zoomSettings['objectFrom']['dblCenterU']) + (dblTo * zoomSettings['objectTo']['dblCenterU'])) - (objectCommon['intWidth'] / 2.0)
    dblShiftV = ((dblFrom * zoomSettings['objectFrom']['dblCenterV']) + (dblTo * zoomSettings['objectTo']['dblCenterV'])) - (objectCommon['intHeight'] / 2.0)
    dblCropWidth = (dblFrom * zoomSettings['objectFrom']['intCropWidth']) + (dblTo * zoomSettings['objectTo']['intCropWidth'])
    dblCropHeight = (dblFrom * zoomSettings['objectFrom']['intCropHeight']) + (dblTo * zoomSettings['objectTo']['intCropHeight'])

    dblDepthFrom = objectCommon['objectDepthrange'][0]
    dblDepthTo = objectCommon['objectDepthrange'][0] * (dblCropWidth / max(zoomSettings['objectFrom']['intCropWidth'], zoomSettings['objectTo']['intCropWidth']))
    # dblDepthTo = objectCommon['objectDepthrange'][0] * (dblCropWidth / zoomSettings['objectFrom']['intCropWidth'])
    _, tensorShift = process_shift({
        'tensorPoints': objectCommon['tensorRawPoints'],
        'dblShiftU': dblShiftU,
        'dblShiftV': dblShiftV,
        'dblDepthFrom': dblDepthFrom,
        'dblDepthTo': dblDepthTo
    }, objectCommon)

    return tensorShift

# Compute disocclusion mask from an image and its disparity maps and zoom settings
def get_masks(tensorImage, tensorDisparity, tensorDepth, zoom_settings, camera, AFromB=True, tensorContext=None):
    masksList = []
    shiftList = []
    objectList = []

    dblFocal = camera['focal']
    dblBaseline = camera['baseline']
    intWidth = tensorImage.shape[3]
    intHeight = tensorImage.shape[2]

    tensorValid = (spatial_filter(tensorDisparity / tensorDisparity.max(), 'laplacian').abs() < 0.03).float()
    tensorPoints = depth_to_points(tensorDepth * tensorValid, dblFocal)


    for idx in range(tensorImage.shape[0]):
        objectCommon = {}
        objectCommon['dblFocal'] = dblFocal
        objectCommon['dblBaseline'] = dblBaseline
        objectCommon['intWidth'] = intWidth
        objectCommon['intHeight'] = intHeight
        objectCommon['tensorRawImage'] = tensorImage[idx]
        objectCommon['tensorRawDisparity'] = tensorDisparity[idx]
        
        objectCommon['dblDispmin'] = tensorDisparity[idx].min().item()
        objectCommon['dblDispmax'] = tensorDisparity[idx].max().item()
        objectCommon['objectDepthrange'] = cv2.minMaxLoc(src=tensorDepth[idx, 0, 128:-128, 128:-128].detach().cpu().numpy(), mask=None)
        objectCommon['tensorRawPoints'] = tensorPoints[idx].view(1, 3, -1)
        objectCommon['zoomSettings'] = get_item_in_dict(zoom_settings, idx) 

        tensorShift = get_tensor_shift(objectCommon)

        shiftList.append(tensorShift)
        objectList.append(objectCommon)

    tensorShift = torch.cat(shiftList)

    if AFromB:
        tensorMasks = generate_mask(tensorPoints.view(tensorImage.shape[0], 3, -1),
                                    tensorShift,
                                    intWidth, intHeight, dblFocal, dblBaseline)
        return tensorMasks, tensorShift, objectList
    else:
        if tensorContext is not None:
            tensorRender, tensorMasks = render_pointcloud(tensorPoints.view(tensorImage.shape[0], 3, -1) + tensorShift,
                                    torch.cat([tensorImage, tensorDisparity, tensorContext], 1).view(tensorImage.shape[0], 68, -1),
                                    intWidth, intHeight, dblFocal, dblBaseline)
        else:
            tensorRender, tensorMasks = render_pointcloud(tensorPoints.view(tensorImage.shape[0], 3, -1) + tensorShift,
                                    torch.cat([tensorImage, tensorDisparity], 1).view(tensorImage.shape[0], 4, -1),
                                    intWidth, intHeight, dblFocal, dblBaseline)
        
        tensorMasks = (tensorMasks > 0.0).float()
        return tensorRender, tensorMasks, tensorPoints.view(tensorImage.shape[0], 3, -1), tensorShift, objectList
    
# Create halfway view C from view A and B
def generate_new_view_from_inpaint(tensorPointsA, 
                                    tensorImageA, 
                                    tensorDisparityA, 
                                    tensorDepthA,
                                    tensorImageB, 
                                    tensorDisparityB, 
                                    tensorDepthB, 
                                    tensorMaskB, 
                                    tensorShift,
                                    camera):
    dblFocal = camera['focal']
    dblBaseline = camera['baseline']
    intHeight, intWidth = tensorImageA.shape[-2:]
    N = tensorImageA.shape[0]
    tensorValidB = (spatial_filter(tensorDisparityB / tensorDisparityB.max(), 'laplacian').abs() < 0.03).float()
    tensorPointsB = depth_to_points(tensorDepthB , dblFocal).view(N,3,-1) - tensorShift

    tensorMaskB = (tensorMaskB == 0.0).view(N, 1, 1, -1)

    # Concat everything into on object for rendering
    lengths = tensorMaskB.view(N, 1, -1).sum(-1).cpu().numpy().reshape(-1)
    maxLength = lengths.max()
    lengthA = tensorPointsA.shape[-1]

    tensorPoints = torch.cat([tensorPointsA, tensorPointsB], 2).view(N,1,3,-1)
    tensorImage = torch.cat([tensorImageA.view(N,3,-1), tensorImageB.view(N,3,-1)],2)
    tensorDepth = torch.cat([tensorDepthA.view(N,1,-1), tensorDepthB.view(N,1,-1)],2)

    tensorRenderCHole, tensorMasksC = render_pointcloud(tensorPoints.view(N, 3, -1) + tensorShift.view(N, 3, 1)/2,
                                torch.cat([tensorImage, tensorDepth], 1).view(N, 4, -1),
                                intWidth, intHeight, dblFocal, dblBaseline)



    return tensorRenderCHole, tensorMasksC

    
# Sample random cropping windows for simulating 3D KBE
def get_random_zoom(imgHeight, imgWidth):
    centerUFrom = np.random.uniform(0.3,0.7) * imgWidth
    centerVFrom = np.random.uniform(0.3,0.7) * imgHeight
    ratioCropUFrom = np.random.uniform(0.6, 2 / imgWidth * min(imgWidth - centerUFrom, centerUFrom))
    ratioCropVFrom = np.random.uniform(0.6, 2 / imgHeight * min(imgHeight - centerVFrom, centerVFrom))
    ratioCropFrom = min(ratioCropUFrom, ratioCropVFrom)

    objectFrom = {
        'dblCenterU': int(centerUFrom),
        'dblCenterV': int(centerVFrom),
        'intCropWidth': int(imgWidth * ratioCropFrom),
        'intCropHeight': int(imgHeight * ratioCropFrom)
    }

    centerUTo = np.random.uniform(max(0.3, centerUFrom/imgWidth - 0.15 * centerUFrom/imgWidth),min(0.7, centerUFrom/imgWidth + 0.15 * centerUFrom/imgWidth)) * imgWidth
    centerVTo = np.random.uniform(max(0.3, centerVFrom/imgHeight - 0.15 * centerVFrom/imgHeight),min(0.7, centerVFrom/imgHeight + 0.15 * centerVFrom/imgHeight)) * imgHeight
    ratioCropUTo = np.random.uniform(0.6, 2 / imgWidth * min(imgWidth - centerUTo, centerUTo))
    ratioCropVTo = np.random.uniform(0.6, 2 / imgHeight * min(imgHeight - centerVTo, centerVTo))
    ratioCropTo = min(ratioCropUTo, ratioCropVTo)

    objectTo = {
        'dblCenterU': int(centerUTo),
        'dblCenterV': int(centerVTo),
        'intCropWidth': int(imgWidth * ratioCropTo),
        'intCropHeight': int(imgHeight * ratioCropTo)
    }

    return objectFrom, objectTo

def get_item_in_dict(dict_in, idx):
    new_dict = {}
    for key, value in dict_in.items():
        if type(value) == dict:
            new_dict[key] = get_item_in_dict(value, idx)
        else:
            new_dict[key] = value[idx]
    return new_dict


###########################################################
# Code from https://https://github.com/NVIDIA/partialconv #
###########################################################

def gram_matrix(input_tensor):
    """
    Compute Gram matrix

    :param input_tensor: input tensor with shape
     (batch_size, nbr_channels, height, width)
    :return: Gram matrix of y
    """
    (b, ch, h, w) = input_tensor.size()
    features = input_tensor.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    
    # more efficient and formal way to avoid underflow for mixed precision training
    input = torch.zeros(b, ch, ch).type(features.type())
    gram = torch.baddbmm(input, features, features_t, beta=0, alpha=1./(ch * h * w), out=None)

    return gram

def make_vgg16_layers(style_avg_pool = False):
    """
    make_vgg16_layers

    Return a custom vgg16 feature module with avg pooling
    """
    vgg16_cfg = [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
        512, 512, 512, 'M', 512, 512, 512, 'M'
    ]

    layers = []
    in_channels = 3
    for v in vgg16_cfg:
        if v == 'M':
            if style_avg_pool:
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class VGG16Partial(nn.Module):
    """
    VGG16 partial model
    """
    def __init__(self, layer_num=3):
        """
        Init

        :param layer_num: number of layers
        """
        super().__init__()
        vgg_model = models.vgg16(pretrained=True)
        vgg_model.features = make_vgg16_layers()

        vgg_pretrained_features = vgg_model.features

        assert layer_num > 0
        assert isinstance(layer_num, int)
        self.layer_num = layer_num

        self.slice1 = torch.nn.Sequential()
        for x in range(5):  # 4
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 1:
            self.slice2 = torch.nn.Sequential()
            for x in range(5, 10):  # (4, 9)
                self.slice2.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 2:
            self.slice3 = torch.nn.Sequential()
            for x in range(10, 17):  # (9, 16)
                self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 3:
            self.slice4 = torch.nn.Sequential()
            for x in range(17, 24):  # (16, 23)
                self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def normalize_batch(batch, div_factor=1.0):
        """
        Normalize batch

        :param batch: input tensor with shape
         (batch_size, nbr_channels, height, width)
        :param div_factor: normalizing factor before data whitening
        :return: normalized data, tensor with shape
         (batch_size, nbr_channels, height, width)
        """
        # normalize using imagenet mean and std
        mean = batch.data.new(batch.data.size())
        std = batch.data.new(batch.data.size())
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        batch = torch.div(batch, div_factor)

        batch -= Variable(mean)
        batch = torch.div(batch, Variable(std))
        return batch

    def forward(self, x):
        """
        Forward, get features used for perceptual loss

        :param x: input tensor with shape
         (batch_size, nbr_channels, height, width)
        :return: list of self.layer_num feature maps used to compute the
         perceptual loss
        """
        h = self.slice1(x)
        h1 = h

        output = []

        if self.layer_num == 1:
            output = [h1]
        elif self.layer_num == 2:
            h = self.slice2(h)
            h2 = h
            output = [h1, h2]
        elif self.layer_num == 3:
            h = self.slice2(h)
            h2 = h
            h = self.slice3(h)
            h3 = h
            output = [h1, h2, h3]
        elif self.layer_num >= 4:
            h = self.slice2(h)
            h2 = h
            h = self.slice3(h)
            h3 = h
            h = self.slice4(h)
            h4 = h
            output = [h1, h2, h3, h4]
        return output

