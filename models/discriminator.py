import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.utils.spectral_norm import remove_spectral_norm
from utils.utils import VGG16Partial, device

class ConvBlock(nn.Module):
    '''(conv => BN => LeakyReLU)'''
    def __init__(self, in_ch, out_ch, stride=2, dilation=1, first=False):
        super(ConvBlock, self).__init__()
        if first:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, dilation=dilation),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, dilation=dilation),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class VGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, small=True):
        super(VGGBlock, self).__init__()
        if small:
            self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
        else:
            self.block = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
    
    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self, channels=None, dilation=None, stride=None):
        super(Discriminator, self).__init__()

        if channels is None:
            self.net = nn.Sequential(ConvBlock(3, 32, first=True),
                                        ConvBlock(32, 64, first=False),
                                        ConvBlock(64, 128, first=False),
                                        ConvBlock(128, 256, first=False))
            self.outConv = nn.Conv2d(in_channels=256, out_channels=1, stride=1, kernel_size=4, padding=1)
        else:           
            blocks = [ConvBlock(channels[i], channels[i+1],
                                                stride=stride[i],
                                                dilation=dilation[i],
                                                first=False) for i in range(len(channels)-1)]
            self.net = nn.Sequential(*blocks)
            self.outConv = nn.Conv2d(in_channels=channels[-1], out_channels=1, stride=1, kernel_size=4, padding=1)

        self.loss = nn.MSELoss()

        self.sigmoid = nn.Sigmoid()

        self.spectral_norm = False
    
    def forward(self, tensorImage):
        tensorImage = self.net(tensorImage)
        return self.outConv(tensorImage)
    
    def adversarialLoss(self, tensorImage, isReal):
        predictions = self.forward(tensorImage)
        
        if isReal:
            labels = torch.ones_like(predictions).to(device)
        else:
            labels = torch.zeros_like(predictions).to(device)

        return self.loss(predictions, labels)
    

class PerceptualDiscriminator(nn.Module):
    def __init__(self):
        super(PerceptualDiscriminator, self).__init__()

        self.extractor = VGG16Partial().eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        self.net = nn.Sequential(ConvBlock(256, 256, first=False),
                                    ConvBlock(256, 256, first=False),
                                    ConvBlock(256, 256, first=False))

        self.outConv = nn.Conv2d(in_channels=256, out_channels=1, stride=1, kernel_size=4, padding=1)

        self.sigmoid = nn.Sigmoid()

        self.loss = nn.MSELoss()

        self.spectral_norm = False
    
    def forward(self, tensorImage):
        vggFeatures = self.extractor(tensorImage)
        tensorImage = self.net(vggFeatures[-1])
        return self.outConv(tensorImage)
    
    def adversarialLoss(self, tensorImage, isReal):
        predictions = self.forward(tensorImage)
        loss = 0
        
        if isReal:
            labels = torch.ones_like(predictions).to(device)
        else:
            labels = torch.zeros_like(predictions).to(device)

        return self.loss(predictions, labels)
        

class MultiScalePerceptualDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScalePerceptualDiscriminator, self).__init__()

        self.extractor = VGG16Partial().eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        self.ConvBlock0 = VGGBlock(3, 64)
        self.ConvBlock1 = VGGBlock(128, 128)
        self.ConvBlock2 = VGGBlock(256, 256, small=False)

        self.localD1 = Discriminator([256, 256, 256], [1, 1], [1, 1])   
        self.localD2 = Discriminator([512, 256, 256], [1, 1], [2, 1])

        self.Dmain = Discriminator([512, 256, 256, 256], [8, 4, 1], [1, 1, 1])

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

        self.spectral_norm = False
    
    def forward(self, tensorImage):
        [vggF1, vggF2, vggF3] = self.extractor(tensorImage)

        F1 = self.ConvBlock0(tensorImage)
        F2 = self.ConvBlock1(torch.cat([vggF1, F1], dim=1))
        F3 = self.ConvBlock2(torch.cat([vggF2, F2], dim=1))

        return [self.sigmoid(self.localD1(torch.cat([vggF2, F2], dim=1))), 
                self.sigmoid(self.localD2(torch.cat([vggF3, F3], dim=1))), 
                self.sigmoid(self.Dmain(torch.cat([vggF3, F3], dim=1)))]
    
    def adversarialLoss(self, tensorImage, isReal):
        predictions = self.forward(tensorImage)
        loss = 0
        for pred in predictions:
            if isReal:
                labels = torch.ones_like(pred).to(device)
            else:
                labels = torch.zeros_like(pred).to(device)

            loss += self.loss(pred, labels)
        
        return loss/len(predictions)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()

        self.ConvBlock0 = VGGBlock(3, 64)
        self.ConvBlock1 = VGGBlock(64, 128)
        self.ConvBlock2 = VGGBlock(128, 256, small=False)

        self.localD1 = Discriminator([128, 256, 256], [1, 1], [1, 1])   
        self.localD2 = Discriminator([256, 256, 256], [1, 1], [2, 1])

        self.Dmain = Discriminator([256, 256, 256, 256], [8, 4, 1], [1, 1, 1])

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

        self.spectral_norm = False
    
    def forward(self, tensorImage):

        F1 = self.ConvBlock0(tensorImage)
        F2 = self.ConvBlock1(F1)
        F3 = self.ConvBlock2(F2)

        return [self.sigmoid(self.localD1(F2)), 
                self.sigmoid(self.localD2(F3)), 
                self.sigmoid(self.Dmain(F3))]
    
    def adversarialLoss(self, tensorImage, isReal):
        predictions = self.forward(tensorImage)
        loss = 0
        for pred in predictions:
            if isReal:
                labels = torch.ones_like(pred).to(device)
            else:
                labels = torch.zeros_like(pred).to(device)

            loss += self.loss(pred, labels)
        
        return loss/len(predictions)



class MPDDiscriminator(nn.Module):
    def __init__(self):
        super(MPDDiscriminator, self).__init__()

        self.extractor = VGG16Partial().eval()
        for p in self.extractor.parameters():
            p.requires_grad = False

        self.ConvBlock0 = VGGBlock(4, 64)
        self.ConvBlock1 = VGGBlock(128, 128)
        self.ConvBlock2 = VGGBlock(256, 256, small=False)

        self.localD1 = Discriminator([256, 256, 256], [1, 1], [1, 1])   
        self.localD2 = Discriminator([512, 256, 256], [1, 1], [2, 1])

        self.Dmain = Discriminator([512, 256, 256, 256], [8, 4, 1], [1, 1, 1])

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

        self.spectral_norm = False
    
    def forward(self, tensorImage, tensorDisparity):
        [vggF1, vggF2, vggF3] = self.extractor(tensorImage)

        F1 = self.ConvBlock0(torch.cat([tensorImage, tensorDisparity], dim=1))
        F2 = self.ConvBlock1(torch.cat([vggF1, F1], dim=1))
        F3 = self.ConvBlock2(torch.cat([vggF2, F2], dim=1))

        return [self.sigmoid(self.localD1(torch.cat([vggF2, F2], dim=1))), 
                self.sigmoid(self.localD2(torch.cat([vggF3, F3], dim=1))), 
                self.sigmoid(self.Dmain(torch.cat([vggF3, F3], dim=1)))]
    
    def adversarialLoss(self, tensorImage, tensorDisparity, isReal):
        predictions = self.forward(tensorImage, tensorDisparity)
        loss = 0
        for pred in predictions:
            if isReal:
                labels = torch.ones_like(pred).to(device)
            else:
                labels = torch.zeros_like(pred).to(device)

            loss += self.loss(pred, labels)
        
        return loss/len(predictions)

