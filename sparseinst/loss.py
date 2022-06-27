import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops

from detectron2.utils.registry import Registry

SPARSE_INST_MATCHER_REGISTRY = Registry("SPARSE_INST_MATCHER")
SPARSE_INST_MATCHER_REGISTRY.__doc__ = "Matcher for SparseInst"
SPARSE_INST_CRITERION_REGISTRY = Registry("SPARSE_INST_CRITERION")
SPARSE_INST_CRITERION_REGISTRY.__doc__ = "Criterion for SparseInst"


class SparseInstMatcher(nn.Cell):
	def __init__(self,cfg):
		super().__init__()
		self.alpha=cfg.MODEL.SPARSE_INST.MATCHER.ALPHA 
		self.beta=cfg.MODEL.SPARSE_INST.MATCHER.BETA 
		self.mask