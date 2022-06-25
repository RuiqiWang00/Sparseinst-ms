from mindvision.classification.models.backbones import ResidualBlock
import mindspore.nn as nn


class Resnet(nn.Cell):
	def __init__(self,layer_nums):
		super(Resnet,self).__init__()
		self.conv1=nn.Conv2d(3,64,7,stride=2)

	def construct(self,x):
		return x

