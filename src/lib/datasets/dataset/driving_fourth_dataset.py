from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
from utils.metrics import get_iou
from lib.datasets.dataset.label_mappings import vehicle_class_groups, bdd_class_groups, vehicle_single_class, vehicle_label_names, combined_label_names, detectron_classes, coco_class_groups, get_remap


import torch.utils.data as data

class DrivingFourth(data.Dataset):
  num_classes = 80
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)


  def modify_json(self, json_file, end_tag, threshold):

    def remap(mp, cls):
      if cls in mp:
        return mp[cls]
      else:
        return -1
    
    def _coco_box_to_bbox(bbox):
      return [(bbox[0]), (bbox[1]),
              (bbox[2] + bbox[0]), (bbox[3] + bbox[1])]
    
    A = json.load(open(json_file, 'r'))
    annots = []
    for x in A['annotations']:
      new_cat_id = 0
      if 'score' in x:  # mmdetect detections
        if x['score'] > threshold:
          new_cat_id = remap(self.remap_coco2bdd, x['category_id']-1)
        else:
          continue
      else: # coco detections
        if x['category_id'] in self._valid_coco_ids:
          new_cat_id = remap(self.remap_coco2bdd, self._valid_coco_ids.index(x['category_id']))
        else:
          continue
      if new_cat_id == -1: # check if it's a bad remap
        continue
      else:
        x['category_id'] = new_cat_id
        annots += [x]

    # create an index to easily get all images in one frame
    imgToAnns = defaultdict(list)
    for ann in annots:
      imgToAnns[ann['image_id']].append(ann)

    leftover_annots = []
    for k, inst in imgToAnns.items():
      if 'score' in inst[0]: # only do this for the frames that are non-coco
        inst = sorted(inst, key=lambda k: k['score'])
        # get rid of overlapping instances using nms
        rejected_list = []
        for i in range(len(inst)):
          for j in range(len(inst)):
            if j in rejected_list: 
              continue
            if i > j and (inst[i]['category_id'] == inst[j]['category_id']):
              iou_calc = get_iou(_coco_box_to_bbox(inst[i]['bbox']), _coco_box_to_bbox(inst[j]['bbox']))
              if iou_calc > self.opt.nms_iou_thresh:
                rejected_list += [i]
        leftover_annots += [i for j, i in enumerate(inst) if j not in rejected_list]
      else:
        leftover_annots += inst
      
    
    A['annotations'] = leftover_annots
    A['images'] = list(filter(lambda x : x['id'] in imgToAnns.keys(),  A['images']))
    json.dump(A, open(f'/scratch/jl5/{end_tag}', 'w'))

  def __init__(self, opt, split):
    super(DrivingFourth, self).__init__()
    self.opt = opt
    self._valid_coco_ids = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
      24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
      37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
      58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
      72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
      82, 84, 85, 86, 87, 88, 89, 90]

    if opt.agg_classes:
      self.num_classes =  len(combined_label_names)
      self.class_name = combined_label_names
      self.remap_coco2bdd = get_remap(coco_class_groups)
    elif opt.single_class:
      self.num_classes = 1
      self.class_name = [opt.single_class]
      self.remap_coco2bdd = get_remap([[detectron_classes.index(opt.single_class)]])
    else:
      self.class_name = ['__ignore__'] + detectron_classes
    
    self.data_dir = '/scratch/jl5/'
    self.img_dir = os.path.join(self.data_dir, 'coco/images/train2017')
    end_tag = ''
    if split == 'test':
      self.annot_path = '/data2/jl5/mmdetect_results/driving1000/fourth1.json'
      end_tag = 'fourth_test.json'
    else:
      self.annot_path = '/data2/jl5/mmdetect_results/driving1000/coco_driving1000_offset.json' # coco_offset_fourth0.json'
      end_tag = 'fourth_train.json'
    self.modify_json(self.annot_path, end_tag, opt.data_thresh)
    self.annot_path = '/scratch/jl5/fourth_train.json'
    
    self.max_objs = 128
    self.cat_ids = {v: i for i, v in enumerate([j for j in range(self.num_classes)])}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    

    print('==> initializing coco 2017 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_coco_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
