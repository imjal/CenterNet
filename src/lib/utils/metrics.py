import numpy as np
from models.decode import ctdet_filt_centers, ctdet_decode
import pdb
from abc import ABC, abstractmethod
import copy
from lib.datasets.dataset.label_mappings import coco_class_groups, get_remap, bdd_class_groups, coco2bdd_class_groups, detectron_classes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1_x1 = bb1[0]
    bb1_x2 = bb1[2]
    bb1_y1 = bb1[1]
    bb1_y2 = bb1[3]

    bb2_x1 = bb2[0]
    bb2_x2 = bb2[2]
    bb2_y1 = bb2[1]
    bb2_y2 = bb2[3]

    assert bb1_x1 < bb1_x2
    assert bb1_y1 < bb1_y2
    assert bb2_x1 < bb2_x2
    assert bb2_y1 < bb2_y2

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1_x1, bb2_x1)
    y_top = max(bb1_y1, bb2_y1)
    x_right = min(bb1_x2, bb2_x2)
    y_bottom = min(bb1_y2, bb2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1)
    bb2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# TODO: find some fix for this to make it less disgusting
def ret_categories():
    return [{'supercategory': 'person', 'id': 1, 'name': 'person'}, 
    {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}, {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
    {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}, {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}, 
    {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'}, {'supercategory': 'vehicle', 'id': 7, 'name': 'train'}, 
    {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}, {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}, 
    {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'}, {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'}, {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}, {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'}, {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'}, {'supercategory': 'animal', 'id': 16, 'name': 'bird'}, {'supercategory': 'animal', 'id': 17, 'name': 'cat'}, {'supercategory': 'animal', 'id': 18, 'name': 'dog'}, {'supercategory': 'animal', 'id': 19, 'name': 'horse'}, {'supercategory': 'animal', 'id': 20, 'name': 'sheep'}, {'supercategory': 'animal', 'id': 21, 'name': 'cow'}, {'supercategory': 'animal', 'id': 22, 'name': 'elephant'}, {'supercategory': 'animal', 'id': 23, 'name': 'bear'}, {'supercategory': 'animal', 'id': 24, 'name': 'zebra'}, {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'}, {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'}, {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'}, {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'}, {'supercategory': 'accessory', 'id': 32, 'name': 'tie'}, {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'}, {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'}, {'supercategory': 'sports', 'id': 35, 'name': 'skis'}, {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'}, {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'}, {'supercategory': 'sports', 'id': 38, 'name': 'kite'}, {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'}, {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'}, {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'}, {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'}, {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'}, {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'}, {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'}, {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'}, {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'}, {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'}, {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'}, {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'}, {'supercategory': 'food', 'id': 52, 'name': 'banana'}, {'supercategory': 'food', 'id': 53, 'name': 'apple'}, {'supercategory': 'food', 'id': 54, 'name': 'sandwich'}, {'supercategory': 'food', 'id': 55, 'name': 'orange'}, {'supercategory': 'food', 'id': 56, 'name': 'broccoli'}, {'supercategory': 'food', 'id': 57, 'name': 'carrot'}, {'supercategory': 'food', 'id': 58, 'name': 'hot dog'}, {'supercategory': 'food', 'id': 59, 'name': 'pizza'}, {'supercategory': 'food', 'id': 60, 'name': 'donut'}, {'supercategory': 'food', 'id': 61, 'name': 'cake'}, {'supercategory': 'furniture', 'id': 62, 'name': 'chair'}, {'supercategory': 'furniture', 'id': 63, 'name': 'couch'}, {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'}, {'supercategory': 'furniture', 'id': 65, 'name': 'bed'}, {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'}, {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'}, {'supercategory': 'electronic', 'id': 72, 'name': 'tv'}, {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'}, {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'}, {'supercategory': 'electronic', 'id': 75, 'name': 'remote'}, {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'}, {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'}, {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'}, {'supercategory': 'appliance', 'id': 79, 'name': 'oven'}, {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'}, {'supercategory': 'appliance', 'id': 81, 'name': 'sink'}, {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'}, {'supercategory': 'indoor', 'id': 84, 'name': 'book'}, {'supercategory': 'indoor', 'id': 85, 'name': 'clock'}, {'supercategory': 'indoor', 'id': 86, 'name': 'vase'}, {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'}, {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}, {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'}, {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}]

def ret_categories_downsized():
    return [{'supercategory': 'person', 'id': 0, 'name': 'person'}, 
    {'supercategory': 'vehicle', 'id': 1, 'name': '4 wheeler'}, {'supercategory': 'vehicle', 'id': 2, 'name': '2 wheeler'}]


class AccumCOCODetResult:
    def __init__(self):
        self.cocoDt = []
        self.dt_counter = 0
        # self.remap_coco2bdd = get_remap(coco2bdd_class_groups)

    def add_det_to_coco(self, iter_id, results_dir):
        '''
        convert a CenterNet detection to coco image instance
        '''
        def remap(mp, cls):
            if cls in mp:
                return mp[cls]
            else:
                return -1
        # fig,ax = plt.subplots(1)
        # im = cv2.imread('/home/jl5/CenterNet/tmp.png')[:, :, ::-1]
        # ax.imshow(im)
        for key, val in results_dir.items():
            if len(val) == 0:
                continue
            for i in range(len(val)):
                det = val[i]
                bbox = [det[0], det[1], det[2]-det[0], det[3]- det[1]]
                res = {
                    "image_id": int(iter_id),
                    "category_id": key-1, # key is the class id
                    "bbox": bbox,
                    "score": det[4],
                    "area": bbox[2] * bbox[3] # box area
                }
                # rect = patches.Rectangle((bbox[0], bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
                # ax.add_patch(rect)
                # if is_baseline:
                #     # remapped = remap(self.remap_coco2bdd, detectron_classes[int(dets[i][5])])
                #     # if remapped < 0:
                #     #     continue
                #     # res["category_id"] = remapped
                #     res['id'] = self.dt_counter
                #     self.dt_counter+=1
                #     self.cocoDt += [res]
                # else:
                res['id'] = self.dt_counter
                self.dt_counter+=1
                self.cocoDt += [res]
        # plt.savefig('/home/jl5/CenterNet/tmp_pic.png')

    def get_dt(self):
        return self.cocoDt



class AccumCOCO:
    def __init__(self):
        self.cocoDt = []
        self.cocoGt = []
        self.gt_counter = 0
        self.dt_counter = 0
        self.remap_coco2bdd = get_remap(coco2bdd_class_groups)

    def add_det_to_coco(self, iter_id, dets, is_gt=False, is_baseline= False):
        '''
        convert a CenterNet detection to coco image instance
        '''
        def remap(mp, cls):
            if cls in mp:
                return mp[cls]
            else:
                return -1
        for i in range(len(dets)):
            bbox = [dets[i][0], dets[i][1], dets[i][2]-dets[i][0], dets[i][3]- dets[i][1]]
            res = {
                "image_id": int(iter_id), 
                "category_id": int(dets[i][5]), 
                "bbox": bbox,
                "score": dets[i][4],
                "area": bbox[2] * bbox[3] # box area
            }
            if is_gt:
                res['iscrowd'] = 0
                res['ignore'] = 0
                res['id'] = self.gt_counter
                self.gt_counter+=1
                self.cocoGt += [res]
            elif is_baseline:
                # remapped = remap(self.remap_coco2bdd, detectron_classes[int(dets[i][5])])
                # if remapped < 0:
                #     continue
                # res["category_id"] = remapped
                res['id'] = self.dt_counter
                self.dt_counter+=1
                self.cocoDt += [res]
            else:
                res['id'] = self.dt_counter
                self.dt_counter+=1
                self.cocoDt += [res]

    def store_metric_coco(self, imgId, batch, output, opt, is_baseline=False):
      dets = ctdet_decode( output['hm'], output['wh'], reg=output['reg'], cat_spec_wh=opt.cat_spec_wh, K=opt.K)
      predictions = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
      predictions[:, :, :4] *= opt.down_ratio * opt.downsample
      dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
      dets_gt = copy.deepcopy(dets_gt)
      dets_gt[:, :, :4] *= opt.down_ratio * opt.downsample
      self.add_det_to_coco(imgId, predictions[0], is_baseline=is_baseline)
      self.add_det_to_coco(imgId, dets_gt[0], is_gt=True)
    
    def get_gt(self):
        return {'annotations': self.cocoGt, 'categories': ret_categories()}
    
    def get_dt(self):
        return self.cocoDt

class Metric(ABC):
    def __init__(self, opt):
        self.opt = opt
    
    @abstractmethod
    def get_score(self, batch, outputs, iter_num):
        pass

class mAP(Metric):
    def __init__(self, opt, center_thresh):
        super().__init__(opt)
        self.center_thresh = center_thresh

    def get_ap(self, recalls, precisions):
        # correct AP calculation
        # first append sentinel values at the end
        recalls = np.concatenate(([0.], recalls, [1.]))
        precisions = np.concatenate(([0.], precisions, [0.]))

        # compute the precision envelope
        for i in range(precisions.size - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(recalls[1:] != recalls[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        return ap

class centermAP(mAP):
    def get_score(self, batch, outputs, iter_id):
        predictions = ctdet_filt_centers(outputs['hm'], outputs['reg']) # is this sorted? 
        gt_inds = np.where(batch['hm'].to("cpu").numpy() == 1.0)
        gt_checked = np.zeros((len(gt_inds)))
        filt_pred = []
        for i in range(len(predictions)):
            if predictions[i][2] >=  self.center_thresh:
                filt_pred += [predictions[i]]

            nd = len(filt_pred)
            tp = np.zeros((nd))
            fp = np.zeros((nd))
            for i, p in enumerate(filt_pred):
                x = p[0]
                y = p[1]
                s = p[2]
                min_dist = -np.inf
                min_arg = -1
                if len(gt_inds) > 0:
                    # dist from each point
                    dist = distance.cdist(gt_inds, np.array([[x, y]]), 'euclidean')
                    # take min & assign
                    min_dist = np.min(dist, axis = 0)
                    min_arg = np.argmin(dist, axis = 0)
                    if gt_checked[min_arg] == 0:
                        tp[i] = 1.
                        gt_checked[jmax, t] = 1
                    else:
                        fp[i] = 1.
        fp = np.cumsum(fp, axis=0)
        tp = np.cumsum(tp, axis=0)
        recalls = tp / float(len(gt_inds))
        precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = get_ap(recalls, precisions)

        return recalls, precisions, ap

class regmAP(mAP):
    def cat_ap(self, filt_pred, gt_boxes, thresholds):
        nd = len(filt_pred)
        gt_checked = np.zeros((len(gt_boxes), len(thresholds)))
        num_gts = len(gt_boxes)
        tp = np.zeros((nd, len(thresholds)))
        fp = np.zeros((nd, len(thresholds)))
        for i, p in enumerate(filt_pred):
            box = p[:4]
            ovmax = -np.inf
            jmax = -1
            if len(gt_boxes) > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(gt_boxes[:, 0], box[0])
                iymin = np.maximum(gt_boxes[:, 1], box[1])
                ixmax = np.minimum(gt_boxes[:, 2], box[2])
                iymax = np.minimum(gt_boxes[:, 3], box[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) +
                        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) *
                        (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            else:
                return None, None, 0
            

            for t, threshold in enumerate(thresholds):
                if ovmax > threshold:
                    if gt_checked[jmax, t] == 0:
                        tp[i, t] = 1.
                        gt_checked[jmax, t] = 1
                    else:
                        fp[i, t] = 1.
                else:
                    fp[i, t] = 1.

        # compute precision recall
        fp = np.cumsum(fp, axis=0)
        tp = np.cumsum(tp, axis=0)
        recalls = tp / float(num_gts)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = np.zeros(len(thresholds))
        for t in range(len(thresholds)):
            ap[t] = self.get_ap(recalls[:, t], precisions[:, t])
        return recalls, precisions, ap

    def get_score(self, batch, output, u, is_baseline=False, save_class_score=False):
        dets = ctdet_decode( output['hm'], output['wh'], reg=output['reg'], cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        predictions = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        predictions[:, :, :4] *= self.opt.down_ratio * self.opt.downsample
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt = copy.deepcopy(dets_gt)
        dets_gt[:, :, :4] *= self.opt.down_ratio * self.opt.downsample

        # debatch
        predictions = predictions[0]
        dets_gt = dets_gt[0]

        thresholds = [0.5, 0.75]
        aps = np.zeros((len(thresholds), self.opt.num_classes))

        num_gts = len(dets_gt)
        image_gt_checked = np.zeros((num_gts, len(thresholds)))
        filt_pred = predictions[predictions[:,4] >= self.center_thresh]
        # if is_baseline: # make instance into a 3 class instance for bdd relabeled gt
        #     d_map = get_remap(coco_class_groups)
        #     pred = filt_pred[:, 5].astype(np.int64)
        #     filt_pred = filt_pred[np.isin(pred, np.array(list(d_map.keys())))]
        #     if len(filt_pred) != 0: 
        #         filt_pred[:, 5] = np.vectorize(d_map.get)(filt_pred[:, 5])

        filt = np.zeros(self.opt.num_classes)
        for i in range(self.opt.num_classes):
            filt_pred_class = filt_pred[filt_pred[:, 5] == i]
            gt_class = dets_gt[dets_gt[:,5] == i]
            filt[i] = (len(gt_class) > 0)
            recalls, precisions, ap = self.cat_ap(filt_pred_class, gt_class, thresholds)
            aps[:, i] = ap
        aps *= 100.0
        nonzero_gt = aps[0][filt != 0]
        mean_aps = np.mean(nonzero_gt, axis=0)
        if save_class_score:
            return mean_aps, aps
        return mean_aps


class segmeanIOU(Metric):
    def get_score(self, batch, outputs, iter_id):
        def convert2d(arr): # convert a one-hot into a thresholded array
            max_arr = arr.max(axis=0)
            new_arr = arr.argmax(axis = 0) + 1
            new_arr[max_arr < 0.1] = 0
            return new_arr
        # add to conf matrix for each image
        pred = convert2d(outputs['seg'].detach().cpu().numpy()[0])
        gt = batch['seg'].detach().cpu().numpy()[0][0]

        N = outputs['seg'].detach().cpu().numpy()[0].shape[0] + 1
        conf_matrix = np.bincount(N * pred.reshape(-1) + gt.reshape(-1), minlength=N ** 2).reshape(N, N)
        
        acc = np.full(N-1, np.nan, dtype=np.float)
        iou = np.full(N-1, np.nan, dtype=np.float)
        tp = conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        class_names = [
            "__ignore__",
            "person", 
            "4-wheeler", 
            "2-wheeler"
        ]
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]
        return miou

class meanIOU(Metric):
    def get_score(self, batch, outputs, iter_id):
        def convert2d(arr): # convert a one-hot into a thresholded array
            max_arr = arr.max(axis=0)
            new_arr = arr.argmax(axis = 0) + 1
            new_arr[max_arr < 0.1] = 0
            return new_arr
        # add to conf matrix for each image
        pred = convert2d(outputs['hm'].detach().cpu().numpy()[0])
        gt = convert2d(batch['hm'].detach().cpu().numpy()[0])
        N = batch['hm'].detach().cpu().numpy()[0].shape[0] + 1
        conf_matrix = np.bincount(N * pred.reshape(-1) + gt.reshape(-1), minlength=N ** 2).reshape(N, N)
        
        acc = np.full(N, np.nan, dtype=np.float)
        iou = np.full(N, np.nan, dtype=np.float)
        tp = conf_matrix.diagonal().astype(np.float)
        pos_gt = np.sum(conf_matrix, axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(conf_matrix, axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        class_names = [
            "__ignore__",
            "person", 
            "4-wheeler", 
            "2-wheeler"
        ]
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]
        return miou

def get_metric(opt):
    if opt.acc_metric == 'mAP':
        metric = regmAP(opt, opt.center_thresh)
    elif opt.acc_metric == 'meanIOU':
        metric = meanIOU(opt)
    return metric