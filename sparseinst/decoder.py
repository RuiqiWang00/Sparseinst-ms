import mindspore
import mindspore.nn as nn
import mindspore.nn.ops as ops


from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d

SPARSE_INST_DECODER_REGISTRY = Registry("SPARSE_INST_DECODER")
SPARSE_INST_DECODER_REGISTRY.__doc__ = "registry for SparseInst decoder"


def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
	convs = []
	for _ in range(num_convs):
		convs.append(
			Conv2d(in_channels, out_channels, 3, has_bias=True))
		convs.append(nn.ReLU())
 		in_channels = out_channels
	return nn.SequentialCell(*convs)


class MaskBranch(nn.Module):

	def __init__(self, cfg, in_channels):
		super().__init__()
		dim = cfg.MODEL.SPARSE_INST.DECODER.MASK.DIM#256
		num_convs = cfg.MODEL.SPARSE_INST.DECODER.MASK.CONVS
		kernel_dim = cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM
		self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
		self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1,has_bias=True)
		self._init_weights()

	def _init_weights(self):
		for m in self.mask_convs.modules():
			if isinstance(m, nn.Conv2d):
				c2_msra_fill(m)
		c2_msra_fill(self.projection)

	def forward(self, features):
 		# mask features (x4 convs)
		features = self.mask_convs(features)
		return self.projection(features)
