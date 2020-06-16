import torch
import torch.nn as nn
import torch.nn.functional as F


class Basic(nn.Module):
	def __init__(self, strType, intChannels):
		super(Basic, self).__init__()

		if strType == 'relu-conv-relu-conv':
			self.moduleMain = nn.Sequential(
				nn.PReLU(num_parameters=intChannels[0], init=0.25),
				nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				nn.PReLU(num_parameters=intChannels[1], init=0.25),
				nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)

		elif strType == 'conv-relu-conv':
			self.moduleMain = nn.Sequential(
				nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
				nn.PReLU(num_parameters=intChannels[1], init=0.25),
				nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
			)


	def forward(self, tensorInput):
		return self.moduleMain(tensorInput)


class Downsample(nn.Module):
	def __init__(self, intChannels):
		super(Downsample, self).__init__()

		self.moduleMain = nn.Sequential(
			nn.PReLU(num_parameters=intChannels[0], init=0.25),
			nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1),
			nn.PReLU(num_parameters=intChannels[1], init=0.25),
			nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tensorInput):
		return self.moduleMain(tensorInput)
	# end
# end

class Upsample(nn.Module):
	def __init__(self, intChannels):
		super(Upsample, self).__init__()

		self.moduleMain = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			nn.PReLU(num_parameters=intChannels[0], init=0.25),
			nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1),
			nn.PReLU(num_parameters=intChannels[1], init=0.25),
			nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1)
		)
	# end

	def forward(self, tensorInput):
		return self.moduleMain(tensorInput)
	# end
# end

class Refine(nn.Module):
	def __init__(self):
		super(Refine, self).__init__()

		self.spectral_norm = False

		self.moduleImageOne = Basic('conv-relu-conv', [ 3, 24, 24 ])
		self.moduleImageTwo = Downsample([ 24, 48, 48 ])
		self.moduleImageThr = Downsample([ 48, 96, 96 ])

		self.moduleDisparityOne = Basic('conv-relu-conv', [ 1, 96, 96 ])
		self.moduleDisparityTwo = Upsample([ 192, 96, 96 ])
		self.moduleDisparityThr = Upsample([ 144, 48, 48 ])
		self.moduleDisparityFou = Basic('conv-relu-conv', [ 72, 24, 24 ])

		self.moduleRefine = Basic('conv-relu-conv', [ 24, 24, 1 ])
	# end

	def forward(self, tensorImage, tensorDisparity):
		tensorMean = [ tensorImage.view(tensorImage.size(0), -1).mean(1, True).view(tensorImage.size(0), 1, 1, 1), tensorDisparity.view(tensorDisparity.size(0), -1).mean(1, True).view(tensorDisparity.size(0), 1, 1, 1) ]
		tensorStd = [ tensorImage.view(tensorImage.size(0), -1).std(1, True).view(tensorImage.size(0), 1, 1, 1), tensorDisparity.view(tensorDisparity.size(0), -1).std(1, True).view(tensorDisparity.size(0), 1, 1, 1) ]

		tensorImage = tensorImage.clone()
		tensorImage -= tensorMean[0]
		tensorImage /= tensorStd[0] + 0.0000001

		tensorDisparity = tensorDisparity.clone()
		tensorDisparity -= tensorMean[1]
		tensorDisparity /= tensorStd[1] + 0.0000001


		tensorImageOne = self.moduleImageOne(tensorImage)
		tensorImage = None
		tensorImageTwo = self.moduleImageTwo(tensorImageOne)
		tensorImageThr = self.moduleImageThr(tensorImageTwo)

		tensorUpsample = self.moduleDisparityOne(tensorDisparity)

		tensorUpsample = self.moduleDisparityTwo(torch.cat([ tensorImageThr, tensorUpsample ], 1)); tensorImageThr = None
		tensorUpsample = self.moduleDisparityThr(torch.cat([ tensorImageTwo, tensorUpsample ], 1)); tensorImageTwo = None
		tensorUpsample = self.moduleDisparityFou(torch.cat([ tensorImageOne, tensorUpsample ], 1)); tensorImageOne = None

		tensorRefine = self.moduleRefine(tensorUpsample)
		tensorRefine *= tensorStd[1] + 0.0000001
		tensorRefine += tensorMean[1]

		return tensorRefine
	# end
# end
