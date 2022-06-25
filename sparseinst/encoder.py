import mindspore.nn as nn
import mindspore.nn.ops as ops


class PyramidPoolingModule(nn.Cell):
	def __init__(self,in_channels,channels=512,sizes=(1,2,3,6)):
		super().__init__()
		self.stages=[]
		self.stages=nn.CellList([self._make_stage(in_channels,channels,size) for size in sizes])
		self.bottleneck=Conv2d(in_channels+len(sizes)*channels,in_channels,1,'pad',has_bias=False)
		self.resize=nn.ResizeBilinear(half_pixel_centers=True)

	def _make_stage(self,features,out_features,size):
		prior=nn.AdaptiveAvgPool2d(output_size=(size,size))
		conv=nn.Conv2d(features,out_features,1,'pad',has_bias=True)
		return nn.SequentialCell(prior,conv)

	def construct(self,feats):
		h, w = feats.shape(2), feats.shape(3)

		prior=[self.resize(ops.ReLU()(stage(feats)),size=(h,w),align_corners=False) for stage in self.stages]+[feats]
		out=ops.ReLU()(self.bottleneck(ops.Concat()(priors,1)))
		return out

class InstanceContextEncoder(nn.Cell):
	def __init__(self,cfg,input_shape):
		super().__init__()
		self.num_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS
		self.in_features = cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES
		# self.norm = cfg.MODEL.SPARSE_INST.ENCODER.NORM
		# depthwise = cfg.MODEL.SPARSE_INST.ENCODER.DEPTHWISE
		self.in_channels = [input_shape[f].channels for f in self.in_features]
		# self.using_bias = self.norm == ""
		fpn_laterals = []
		fpn_outputs = []
		# groups = self.num_channels if depthwise else 1
		for in_channel in reversed(self.in_channels):
			lateral_conv = Conv2d(in_channel, self.num_channels, 1)
			output_conv = Conv2d(self.num_channels, self.num_channels, 3, padding=1)
			c2_xavier_fill(lateral_conv)
			c2_xavier_fill(output_conv)
			fpn_laterals.append(lateral_conv)
			fpn_outputs.append(output_conv)
		self.fpn_laterals = nn.ModuleList(fpn_laterals)
		self.fpn_outputs = nn.ModuleList(fpn_outputs)
		# ppm
		self.ppm = PyramidPoolingModule(self.num_channels, self.num_channels // 4)
		# final fusion
		self.fusion = nn.Conv2d(self.num_channels * 3, self.num_channels, 1)
		c2_msra_fill(self.fusion)