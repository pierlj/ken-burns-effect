import torch
import torch.nn as nn 
import kornia

from utils.utils import derivative_scale, gram_matrix, total_variation_loss, VGG16Partial, device


def compute_loss_ord(disparity, target, mask, mode='L1'):

    if mode == 'L1':
        L1loss = nn.L1Loss(reduction='sum')
        # loss = nn.L1Loss(reduction='none')

        loss = 0
        N = torch.sum(mask)
        
        if N != 0:
            loss = L1loss(disparity * mask, target*mask) / N
        return loss
    elif mode =='rmse':
        Ri = (disparity - target) * mask
        N = torch.sum(mask)
        rmse = 0
        if N != 0:
            rmse = 1/N * (torch.sum(Ri ** 2)) - (1/N * torch.sum(Ri)) ** 2
        # rmse = 1/n * (sum_item(Ri ** 2)) - (1/n * sum_item(Ri)) ** 2
        return rmse
    elif mode =='logrmse':
        Ri = torch.log10(disparity * mask + 1e-7) - torch.log10(target * mask + 1e-7)
        N = torch.sum(mask)
        # n = sum_item(mask)
        logrmse = 0
        if N != 0:
            logrmse = 1/N * (torch.sum(Ri ** 2)) - (0.5/N * torch.sum(Ri)) ** 2
        # logrmse = 1/n * (sum_item(Ri ** 2)) - (0.5/n * sum_item(Ri)) ** 2
        return logrmse
    # return loss(disparity, target)

def compute_loss_grad(disparity, target, mask):        
    scales = [2**i for i in range(4)]
    MSELoss = torch.nn.MSELoss(reduction='sum')
    # MSELoss = torch.nn.MSELoss(reduction='none')
    loss = 0
    for h in scales:
        g_h_disparity_x, g_h_disparity_y = derivative_scale(disparity, h, norm=True)
        g_h_target_x, g_h_target_y = derivative_scale(target, h, norm=True)
        N = torch.sum(mask)
        if N != 0:
            loss += MSELoss(g_h_disparity_x * mask, g_h_target_x * mask) / N
            loss += MSELoss(g_h_disparity_y * mask, g_h_target_y * mask) / N
        # loss += sum_item(MSELoss(g_h_disparity_x * mask, g_h_target_x * mask)) / sum_item(mask)
        # loss += sum_item(MSELoss(g_h_disparity_y * mask, g_h_target_y * mask)) / sum_item(mask)

    return loss

def compute_masked_grad_loss(disparity, masks, scales, kappa):
    lossFunction = torch.nn.L1Loss(reduction='sum')
    loss = 0

    for h in scales:
        g_h_disparity_x, g_h_disparity_y = derivative_scale(disparity, h, norm=False)
        # g_h_disparity_x = torch.nn.functional.pad(g_h_disparity_x, (0, 0, h, 0))
        # g_h_disparity_y = torch.nn.functional.pad(g_h_disparity_y, (h, 0, 0, 0))
        N = torch.sum(masks)
        if N != 0:
            loss += lossFunction(g_h_disparity_x * masks, kappa * masks) / N 
            loss += lossFunction(g_h_disparity_y * masks, kappa * masks) / N
    return loss

class JoinEdgeLoss():
    def __init__(self):
        self.sobel = kornia.filters.Sobel()
        self.gray = kornia.color.RgbToGrayscale()
        self.loss = nn.L1Loss()

    def compute(self, tensorImage, tensorDisparity, tensorMasksExtended):
        edgeImg = (self.sobel(self.gray(tensorImage))>0.1).float()
        edgeDisp = (self.sobel(tensorDisparity)>0.3).float()

        # return self.loss(edgeImg * tensorMasksExtended, edgeDisp * tensorMasksExtended)
        return torch.sum(edgeImg * tensorMasksExtended * (1 - edgeDisp))/torch.sum(tensorMasksExtended)


################################################################################
# Code from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/ #
################################################################################

class InpaintingLoss(nn.Module):
    def __init__(self, kbe_only=False, perceptual=True):
        super().__init__()
        self.l1 = nn.L1Loss()
        
        if perceptual:
            self.extractor = VGG16Partial().to(device).eval()
#         self.extractor = extractor
        self.kbe_only=kbe_only
        self.gaussFilter = kornia.filters.GaussianBlur2d((13, 13), (1.5, 1.5)).to(device)
        self.gaussFilterEdge = kornia.filters.GaussianBlur2d((7, 7), (1, 1)).to(device)
        self.sobel = kornia.filters.Sobel().to(device)
        self.gray = kornia.color.RgbToGrayscale().to(device)
        
        self.joint_edge_loss = JoinEdgeLoss()

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        
        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        if self.kbe_only:
            loss_dict['color'] = self.l1(output, gt)
        else:
            loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
            loss_dict['valid'] = self.l1(mask * output, mask * gt)

            loss_dict['style'] = 0.0
            for i in range(3):
                loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                            gram_matrix(feat_gt[i]))
                loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                            gram_matrix(feat_gt[i]))

            loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict
    
    def forward_adv(self, input, mask, output, disparity=None, disparityGT=False):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['valid'] = self.l1(mask * output, mask * input)

        loss_dict['tv'] = total_variation_loss(output_comp)

        if disparity is not None:
            extendedMasks = (self.gaussFilter(mask) < 1).float()
            edgeImg = (self.sobel(self.gray(output)) > 0.1).float()
            extendedEdges = (self.gaussFilterEdge(edgeImg) > 0).float()
            loss_dict['mask'] = compute_masked_grad_loss(disparity, extendedMasks * (1 - extendedEdges), [1], 0.5)
            # loss_dict['joint_edge'] = self.joint_edge_loss.compute(output.detach(), disparity, mask)
            if disparityGT is not None:
                 loss_dict['valid_depth'] = self.l1(mask * disparity, mask * disparityGT)

        return loss_dict