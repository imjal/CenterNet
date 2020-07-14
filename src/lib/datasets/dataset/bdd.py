from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import torch
import json
import os
import cv2
import pickle
import skvideo.io
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.distillation_utils import batch_segmentation_masks

import math
import pdb
from lib.datasets.dataset.eval_bdd import evaluate_detection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import time

from lib.datasets.dataset.label_mappings import coco2bdd_class_groups, bdd_class_groups, combined_label_names, detectron_classes, coco_class_groups, get_remap

import torch.utils.data as data

class BDD(data.Dataset):
  num_classes = 10
  default_resolution = [736, 1280]
  mean = np.array([0.485, 0.456, 0.406],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)
  
  def __init__(self, opt, split):
    super(BDD, self).__init__()
    opt.data_dir = '/data2/jl5/'
    self.data_dir = os.path.join(opt.data_dir, 'bdd100k')
    self.img_dir = os.path.join(self.data_dir, 'images/100k')
    _ann_name = {'train': 'train', 'val': 'val'}
    self.annot_path = os.path.join(
      self.data_dir, 'labels/coco/', 
      'bdd100k_labels_images_det_coco_{}.json').format(_ann_name[split])
    self.img_dir = os.path.join(
      self.img_dir, '{}').format(_ann_name[split])
    self.max_objs = 50
    self.class_name = [
        "_ignore_",
        "person",
        "rider",
        "car",
        "bus",
        "truck",
        "bike",
        "motor",
        "traffic light",
        "traffic sign",
        "train"
    ]

    self._valid_ids = np.arange(1, self.num_classes + 1, dtype=np.int32)
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt
    

    print('==> initializing BDD {} data.'.format(_ann_name[split]))
    self.coco = coco.COCO(self.annot_path)
    self.images = sorted(self.coco.getImgIds())
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    detections = []
    map_id = all_bboxes['map_img_id']
    del all_bboxes['map_img_id']
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))
          
          detection = {
              "name": map_id[image_id][0],
              "category": self.class_name[int(category_id)],
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples # return 1 

  def save_results(self, results, save_dir):
    ret = self.convert_eval_format(results)
    json.dump(ret, open('{}/results.json'.format(save_dir), 'w'))
    return ret

  def run_eval(self, results, save_dir):
    formatted = self.save_results(results, save_dir)
    gt = json.load(open(f"/data2/jl5/bdd100k/labels/det_gt/val.json", "r"))
    evaluate_detection(gt, formatted, self.opt.center_thresh)



class BDDStream(data.IterableDataset):
  num_classes = 3
  default_resolution = [512, 512]
  mean = np.array([0.485, 0.456, 0.406],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)
  
  def __init__(self, opt, split):
    super(BDDStream, self).__init__()
    opt.data_dir = '/data2/jl5/'
    _ann_name = {'train': 'train'}
    self.max_objs = 50
    self.class_name = ['__ignore__'] + combined_label_names
    self._valid_ids = np.arange(1, self.num_classes + 1, dtype=np.int32)
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    self.split = split
    self.opt = opt
    if not opt.adaptive:
      self.num_classes = 80
    self.video_paths = opt.vid_paths
    self.num_videos = len(opt.vid_paths)
    self.annotation_path = opt.ann_paths
    self.vid_i = 0
    self.length = None
    self.count = 0
    self.loop = False # make into variable later

    self.cap = None
    self.rate = None
    self.width = 1280
    self.height = 720
    
    self.remap_coco2bdd = get_remap(coco2bdd_class_groups)
  
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
  
  def pred_to_inst(self, pred):
    inst = []
    bbox = pred['boxes']
    classes = pred['classes']
    masks = pred['masks']

    def _bbox_to_coco_bbox(bbox):
      return [(bbox[0]), (bbox[1]),
              (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

    def remap(mp, cls):
      if cls in mp:
        return mp[cls] + 1
      else:
        return 0

    for i in range(len(bbox)):
      if remap(self.remap_coco2bdd, detectron_classes[classes[i]]) == 0:
        continue
      ann = {
        'category_id': remap(self.remap_coco2bdd, detectron_classes[classes[i]]),
        'bbox': bbox[i] # _bbox_to_coco_bbox(bbox[i])
        }
      inst.append(ann)
    return inst

  def _frame_from_video(self, video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

  def __next__(self):
    load_vid_time, img_transform_time, create_heatmap_time = 0, 0, 0
    start = time.time()
    if self.cap is None or self.count >= self.length:
      if self.cap is not None and self.vid_i == self.num_videos and self.loop:
        self.vid_i = 0
      elif self.cap is not None and self.vid_i == self.num_videos:
        raise StopIteration
      if self.opt.vidstream == 'skvideo':
        self.cap = skvideo.io.vread(self.video_paths[self.vid_i])
        metadata = skvideo.io.ffprobe(self.video_paths[self.vid_i])
        fr_lst = metadata['video']['@avg_frame_rate'].split('/')
        self.rate = int(fr_lst[0])/int(fr_lst[1])
        self.length = int(metadata['video']['@nb_frames'])
      else:
        self.cap = cv2.VideoCapture(self.video_paths[self.vid_i])
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_gen = self._frame_from_video(self.cap)
      
      self.detections = pickle.load(open(self.annotation_path[self.vid_i], 'rb'))
      self.num_frames = len(self.detections)
      self.count = 0
      self.vid_i +=1
    end_load_vid = time.time()
    load_vid_time = end_load_vid - start

    # load image depending on stream
    start_resize = time.time()
    if self.opt.vidstream == 'skvideo':
      img = self.cap[self.count]
    else:
      original_img = next(self.frame_gen)
      img = cv2.resize(original_img, (1280, 720))

    start_img_transform = time.time()
    anns = self.pred_to_inst(self.detections[self.count])
    num_objs = min(len(anns), self.max_objs)

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        c[0] =  width - c[0] - 1

      # send to gpu
      trans_input = get_affine_transform(
        c, s, 0, [input_w, input_h])
      inp = cv2.warpAffine(img, trans_input,
                          (input_w, input_h),
                          flags=cv2.INTER_LINEAR)
      inp = torch.from_numpy(inp).cuda()
      inp = (inp.float() / 255.)
      
      # if self.split == 'train' and not self.opt.no_color_aug:
      #   color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
      
      inp = (inp - torch.from_numpy(self.mean).cuda()) / torch.from_numpy(self.std).cuda()
      inp = inp.permute(2, 0, 1)

    end_img_transform = time.time()
    img_transform_time = end_img_transform - start_img_transform


    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
    cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian

    gt_det = []
      
    def show_bbox(im):
      fig,ax = plt.subplots(1)
      ax.imshow(im)
      for i in range(num_objs):
        bbox = np.array(anns[i]['bbox'], dtype=np.int32)
        bbox = bbox / 3
        rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2]-bbox[0],bbox[3]-bbox[1],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
      plt.savefig('/home/jl5/CenterNet/tmp.png')
      pdb.set_trace()


    detect = self.detections[self.count]
    if self.opt.task == 'ctdet_semseg':
      seg_mask, weight_mask = batch_segmentation_masks(1, (720*3, 1280 * 3), np.array([detect['boxes']]), np.array([detect['classes']]), detect['masks'],
          np.array([detect['scores']]), [len(detect['boxes'])], True, coco_class_groups, mask_threshold=0.5, box_threshold=self.opt.center_thresh, scale_boxes=False)
      unbatch_seg = seg_mask[0].astype(np.uint8)
      unbatch_weight = weight_mask[0].astype(np.uint8)
      seg_mask = np.expand_dims(cv2.resize(unbatch_seg, (1280, 736)), axis=0).astype(np.int32)
      weight_mask = np.expand_dims(cv2.resize(unbatch_weight, (1280, 736)), axis = 0).astype(bool)

    
    start_detect = time.time()

    for k in range(num_objs):
      ann = anns[k]
      bbox = np.array(ann['bbox'], dtype=np.float32) # self._coco_box_to_bbox(ann['bbox'])
      bbox = bbox / 3 # if need to downsample 
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct_int, radius)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([ct[0] - w / 2, ct[1] - h / 2, 
                      ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
    if self.opt.task == 'ctdet_semseg':
          ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'seg': seg_mask, 'weight_seg': weight_mask}
    else:
      ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
    if self.opt.dense_wh:
      hm_a = hm.max(axis=0, keepdims=True)
      dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
      ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
      del ret['wh']
    elif self.opt.cat_spec_wh:
      ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
      del ret['wh']
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not self.split == 'train':
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
              np.zeros((1, 6), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': self.count}
      ret['meta'] = meta
    self.count+=1

    end_detect_time = time.time()
    create_heatmap_time = end_detect_time - start_detect
    # print("load vid {:.4f} | i,mg transform {:.4f} | create instance {:.4f} \n".format(load_vid_time, img_transform_time, create_heatmap_time))
    return ret
  
  def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
  
  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    detections = [[[] for __ in range(self.num_samples)] \
                  for _ in range(self.num_classes + 1)]
    return detections

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
              open('{}/results.json'.format(save_dir), 'w'))

  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
  
  def __iter__(self):
    return self


    
