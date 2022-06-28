import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from resnet import build_resnet50
from encoder import InstanceContextEncoder
from decoder import GroupIAMDecoder


__all__=["SparseInst"]


class SparseInst(nn.Cell):
	def __init__(self,cfg):
		super().__init__()

		self.backbone=build_resnet50()
		self.encoder=InstanceContextEncoder(cfg,self.backbone.output_channel())
		self.decoder=GroupIAMDecoder(cfg)

		self.pixel_mean=cfg.MODEL.PIXEL_MEAN
		self.pixel_std=cfg.MODEL.PIXEL_STD

		self.cls_threshold = Tensor(cfg.MODEL.SPARSE_INST.CLS_THRESHOLD)
		self.mask_threshold = cfg.MODEL.SPARSE_INST.MASK_THRESHOLD
		self.max_detections = cfg.MODEL.SPARSE_INST.MAX_DETECTIONS

		def normalizer(self, image):
			for i in range(image.shape[1]):
				image[:,i,:,:]=(image[:,i,:,:]-self.pixel_mean[i])/self.pixel_std[i]
			return image

		def padding(self,image,size_divisibility=32,pad_value=0.0):
			h,w=image.shape[2],image.shape[3]

			def g(x):
				new_x=(x//size_divisibility+1)*size_divisibility
				l=(new_x-x)//2
				r=new_x-x-l
				return l,r
			left,right=g(w)
			top,bottom=g(h)

			return ops.Pad(((0,0),(0,0),(top,bottom),(left,right)))(image)

		def preprocess_inputs(self,batched_inputs):
			images=self.padding(self.normalizer(batched_inputs))
			return images

		def construct(self,batched_inputs):

			#input :Tensor(N,C,H,W)
			#output = {
			#"pred_logits": pred_logits,
			#"pred_masks": pred_masks,
			#"pred_scores": pred_scores,
		#}

			images=preprocess_inputs(batched_inputs)
			features=self.backbone(images)
			features=self.encoder(features)
			output=self.decoder(features)

			return output


