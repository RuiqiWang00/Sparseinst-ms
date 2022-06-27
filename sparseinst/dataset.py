import mindspore
from mindspore import Tensor
import mindspore.dataset as ds
from mindspore.dataset.transforms import c_transforms

def build_coco_dataset(cfg,is_train:bool=True):
	coco_dataset_dir=cfg.DATASET_DIR 
	coco_annotation_file=cfg.ANNOTATION_FILE 

	dataset=ds.CocoDataset(dataset_dir=coco_dataset_dir,
				annotation_file=coco_annotation_file,
				task='Stuff')

	#to do :transforms
	return dataset
