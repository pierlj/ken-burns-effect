import torch
import torch.nn as nn
import torch.nn.functional as F

def disparity_adjustment(tensorImage, tensorDisparity, objectPredictions):
	# assert(tensorImage.size(0) == 1)
	# assert(tensorDisparity.size(0) == 1)
	if tensorImage.size(0) != 1:
		tensorImage = tensorImage.expand(1, *tensorImage.size())
		tensorDisparity = tensorDisparity.expand(1, *tensorDisparity.size())
		
	boolUsed = {}
	tensorMasks = []

	for intMask in range(objectPredictions['masks'].size(0)):
		if intMask in boolUsed:
			continue

		elif objectPredictions['scores'][intMask].item() < 0.7:
			continue

		elif objectPredictions['labels'][intMask].item() not in [1, 3, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 ]:
			continue

		# end

		boolUsed[intMask] = True
		tensorMask = (objectPredictions['masks'][(intMask + 0):(intMask + 1), :, :, :] > 0.5).float()

		if tensorMask.sum().item() < 64:
			continue
		# end

		for intMerge in range(objectPredictions['masks'].size(0)):
			if intMerge in boolUsed:
				continue

			elif objectPredictions['scores'][intMerge].item() < 0.7:
				continue

			elif objectPredictions['labels'][intMerge].item() not in [ 2, 4, 27, 28, 31, 32, 33 ]:
				continue

			# end

			tensorMerge = (objectPredictions['masks'][(intMerge + 0):(intMerge + 1), :, :, :] > 0.5).float()

			if ((tensorMask + tensorMerge) > 1.0).sum().item() < 0.03 * tensorMerge.sum().item(): # if masks overlap too little (less than 3% do not merge masks)
				continue
			# end

			boolUsed[intMerge] = True
			tensorMask = (tensorMask + tensorMerge).clamp(0.0, 1.0)
		# end

		tensorMasks.append(tensorMask)
	# end

	tensorAdjusted = F.interpolate(input=tensorDisparity, size=(tensorImage.size(2), tensorImage.size(3)), mode='bilinear', align_corners=False)

	for tensorAdjust in tensorMasks:
		tensorPlane = tensorAdjusted * tensorAdjust

		tensorPlane = F.max_pool2d(input=tensorPlane.neg(), kernel_size=3, stride=1, padding=1).neg()
		tensorPlane = F.max_pool2d(input=tensorPlane.neg(), kernel_size=3, stride=1, padding=1).neg()
		try:
			intLeft = (tensorPlane.sum(2, True) > 0.0).flatten().nonzero()[0].item()
			intTop = (tensorPlane.sum(3, True) > 0.0).flatten().nonzero()[0].item()
			intRight = (tensorPlane.sum(2, True) > 0.0).flatten().nonzero()[-1].item()
			intBottom = (tensorPlane.sum(3, True) > 0.0).flatten().nonzero()[-1].item()

			tensorAdjusted = ((1.0 - tensorAdjust) * tensorAdjusted) + (tensorAdjust * tensorPlane[:, :, int(round(intTop + (0.97 * (intBottom - intTop)))):, :].max())
		except:
			pass
	# end


	return F.interpolate(input=tensorAdjusted, size=(tensorDisparity.size(2), tensorDisparity.size(3)), mode='bilinear', align_corners=False)
# end