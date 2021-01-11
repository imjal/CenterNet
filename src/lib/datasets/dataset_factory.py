from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.driving_fifth_dataset import Driving
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.bdd import BDD, BDDStream
from .dataset.driving_fourth_dataset import DrivingFourth
from .dataset.driving_half_dataset import DrivingHalf


dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP, 
  'bdd': BDD, 
  'fifth': Driving,
  'fourth': DrivingFourth, 
  'half': DrivingHalf
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):
  if dataset == 'bddstream':
    return BDDStream
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
