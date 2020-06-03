from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import cv2
DEBUG = False
import os
import pdb
from PIL import Image


def _bbox_to_coco_bbox(bbox):
  return [(bbox['x1']), (bbox['y1']),
          (bbox['x2'] - bbox['x1']), (bbox['y2'] - bbox['y1'])]

cats = [
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck"
]

cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}

cat_info = []
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i})

splits = ['train', 'val']
path_to_annotations = '/data2/jl5/bdd100k/labels'

for split in splits:
    ret = {'images': [], 'annotations': [], "categories": cat_info}
    ann_set = json.load(open(os.path.join(path_to_annotations, 'bdd100k_labels_images_{}.json'.format(split)), 'r'))
    img_path = f'/data2/jl5/bdd100k/images/100k/{split}/'
    for img in os.listdir(img_path):
        im = Image.open(os.path.join(img_path, img))
        width, height = im.size
        im = {
            'file_name': img, 
            'height': height,
            'width': width,
            'id': img[:-4]
        }
        ret['images'].

    for ann_ind, ann_dict in enumerate(ann_set):
        img_id = ann_dict['name'][:-4]
        ann_labels = ann_dict['labels']
        
        for i in range(len(ann_labels)):
            if ann_labels[i]['category'] == 'drivable area' or ann_labels[i]['category'] == 'lane':
                continue
            ann = {'image_id': img_id,
                'id': ann_labels[i]['id'],
                'category_id': cats.index(ann_labels[i]['category']),
                'bbox': _bbox_to_coco_bbox(ann_labels[i]['box2d']),
                'crowd': 0}
            ret['annotations'].append(ann)
        if DEBUG:
            pdb.set_trace()
        

    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    out_path = '{}/bdd100k_labels_images_coco_{}.json'.format(path_to_annotations, split)
    json.dump(ret, open(out_path, 'w'))

