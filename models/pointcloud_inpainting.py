import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.common import spatial_filter, depth_to_points, render_pointcloud

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

		# end

		if intChannels[0] == intChannels[2]:
			self.moduleShortcut = None

		elif intChannels[0] != intChannels[2]:
			self.moduleShortcut = nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0)

		# end
	# end

	def forward(self, tensorInput):
		if self.moduleShortcut is None:
			return self.moduleMain(tensorInput) + tensorInput

		elif self.moduleShortcut is not None:
			return self.moduleMain(tensorInput) + self.moduleShortcut(tensorInput)

		# end
	# end
# end

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

class Inpaint(nn.Module):
	def __init__(self):
		super(Inpaint, self).__init__()

		self.spectral_norm = False

		self.moduleContext = nn.Sequential(
			nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
			nn.PReLU(num_parameters=64, init=0.25),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
			nn.PReLU(num_parameters=64, init=0.25)
		)

		# tensorImage :: tensorDisparity :: tensorContext (64ch):: tensorMasks
		self.moduleInput = Basic('conv-relu-conv', [ 3 + 1 + 64 + 1, 32, 32 ]) 


		for intRow, intFeatures in [ (0, 32), (1, 64), (2, 128), (3, 256) ]:
			self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x1' + ' - ' + str(intRow) + 'x2', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x2' + ' - ' + str(intRow) + 'x3', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
		# end

		for intCol in [ 0, 1 ]:
			self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([ 32, 64, 64 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([ 64, 128, 128 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([ 128, 256, 256 ]))
		# end

		for intCol in [ 2, 3 ]:
			self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([ 256, 128, 128 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([ 128, 64, 64 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([ 64, 32, 32 ]))
		# end

		self.moduleImage = Basic('conv-relu-conv', [ 32, 32, 3 ])
		self.moduleDisparity = Basic('conv-relu-conv', [ 32, 32, 1 ])
	# end

	def forward(self, tensorMasks, tensorImage=None, tensorDisparity=None, tensorData=None, tensorContext=None):

		if tensorImage is not None and tensorContext is None:
			tensorImage, tensorDisparity = self.normalize_images_disp(tensorImage, tensorDisparity, not_normed=True)

		if tensorData is None and tensorContext is not None:
			tensorData = torch.cat([tensorImage, tensorDisparity, tensorContext], 1)
		elif tensorData is None:
			tensorContext = self.moduleContext(torch.cat([ tensorImage, tensorDisparity ], 1))
			tensorData = torch.cat([tensorImage, tensorDisparity, tensorContext], 1)
		
		tensorColumn = [ None, None, None, None ]
		
		tensorColumn[0] = self.moduleInput(torch.cat([ tensorData, tensorMasks ], 1))
		tensorColumn[1] = self._modules['0x0 - 1x0'](tensorColumn[0])
		tensorColumn[2] = self._modules['1x0 - 2x0'](tensorColumn[1])
		tensorColumn[3] = self._modules['2x0 - 3x0'](tensorColumn[2])

		intColumn = 1
		for intRow in range(len(tensorColumn)):
			tensorColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow])
			if intRow != 0:
				tensorColumn[intRow] += self._modules[str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow - 1])
			# end
		# end

		intColumn = 2
		for intRow in range(len(tensorColumn) -1, -1, -1):
			tensorColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow])
			if intRow != len(tensorColumn) - 1:
				tensorUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow + 1])

				if tensorUp.size(2) != tensorColumn[intRow].size(2): tensorUp = F.pad(input=tensorUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tensorUp.size(3) != tensorColumn[intRow].size(3): tensorUp = F.pad(input=tensorUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tensorColumn[intRow] += tensorUp
			# end
		# end

		intColumn = 3
		for intRow in range(len(tensorColumn) -1, -1, -1):
			tensorColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow])
			if intRow != len(tensorColumn) - 1:
				tensorUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tensorColumn[intRow + 1])

				if tensorUp.size(2) != tensorColumn[intRow].size(2): tensorUp = F.pad(input=tensorUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tensorUp.size(3) != tensorColumn[intRow].size(3): tensorUp = F.pad(input=tensorUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tensorColumn[intRow] += tensorUp
			# end
		# end
		tensorImage = self.moduleImage(tensorColumn[0])
		tensorDisparity = self.moduleDisparity(tensorColumn[0])
		tensorImage, tensorDisparity = self.normalize_images_disp(tensorImage, tensorDisparity, not_normed=False)

		return {
			'tensorExisting': tensorMasks,
			'tensorImage': tensorImage.clamp(0.0, 1.0) if self.training == False else tensorImage,
			'tensorDisparity': F.threshold(input=tensorDisparity, threshold=0.0, value=0.0)
		}
	# end
# end

	def pointcloud_inpainting(self, tensorImage, tensorDisparity, tensorShift, objectCommon, dblFocal=None):

		if dblFocal is None:
			dblFocal = objectCommon['dblFocal']

		assert tensorImage.shape[0] == 1, 'Please process one image at a time.'
		# Compute detpth and point cloud
		tensorDepth = (dblFocal * objectCommon['dblBaseline']) / (tensorDisparity + 0.0000001)
		tensorValid = (spatial_filter(tensorDisparity / tensorDisparity.max(), 'laplacian').abs() < 0.03).float()
		tensorPoints = depth_to_points(tensorDepth * tensorValid, dblFocal)
		tensorPoints = tensorPoints.view(1, 3, -1)

		tensorImage, tensorDisparity = self.normalize_images_disp(tensorImage, tensorDisparity, not_normed=True)

		tensorContext = self.moduleContext(torch.cat([ tensorImage, tensorDisparity ], 1))

		tensorRender, tensorExisting = render_pointcloud(tensorPoints + tensorShift, 
								torch.cat([ tensorImage, tensorDisparity, tensorContext ], 1).view(1, 68, -1), 
								objectCommon['intWidth'], 
								objectCommon['intHeight'], 
								dblFocal, 
								objectCommon['dblBaseline'])

		tensorExisting = (tensorExisting > 0.0).float()
		tensorExisting = tensorExisting * spatial_filter(tensorExisting, 'median-5')
		tensorRender = tensorRender * tensorExisting.clone().detach()

		return self.forward(tensorData = tensorRender, tensorMasks=tensorExisting)
	# end

	## When not_normed is true, means and std are computed and tensors are normed
	## When False, un-normed the tensor with existing means/std
	def normalize_images_disp(self, tensorImage, tensorDisparity, not_normed=True):
		if not_normed:
			self.tensorMean = [ tensorImage.view(tensorImage.size(0), -1).mean(1, True).view(tensorImage.size(0), 1, 1, 1), tensorDisparity.view(tensorDisparity.size(0), -1).mean(1, True).view(tensorDisparity.size(0), 1, 1, 1) ]
			self.tensorStd = [ tensorImage.view(tensorImage.size(0), -1).std(1, True).view(tensorImage.size(0), 1, 1, 1), tensorDisparity.view(tensorDisparity.size(0), -1).std(1, True).view(tensorDisparity.size(0), 1, 1, 1) ]

			tensorImage = tensorImage.clone()
			tensorImage -= self.tensorMean[0]
			tensorImage /= self.tensorStd[0] + 0.0000001

			tensorDisparity = tensorDisparity.clone()
			tensorDisparity -= self.tensorMean[1]
			tensorDisparity /= self.tensorStd[1] + 0.0000001
		else:			
			tensorImage *= self.tensorStd[0] + 0.0000001
			tensorImage += self.tensorMean[0]

			tensorDisparity *= self.tensorStd[1] + 0.0000001
			tensorDisparity += self.tensorMean[1]
		
		return tensorImage, tensorDisparity