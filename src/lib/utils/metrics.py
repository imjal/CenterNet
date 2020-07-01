import numpy as np
from models.decode import ctdet_filt_centers, ctdet_decode
import pdb
from abc import ABC, abstractmethod
import copy

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
    def get_score(self, batch, outputs):
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

    def get_score(self, batch, output, center_thresh, u):
        dets = ctdet_decode( output['hm'], output['wh'], reg=output['reg'], cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        predictions = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        predictions[:, :, :4] *= self.opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        
        dets_gt = copy.deepcopy(dets_gt[:, :, :4] * self.opt.down_ratio)

        # debatch
        predictions = predictions[0]
        dets_gt = dets_gt[0]

        thresholds = [0.5, 0.75]
        aps = np.zeros((len(thresholds), self.opt.num_classes))

        num_gts = len(dets_gt)
        image_gt_checked = np.zeros((num_gts, len(thresholds)))
        filt_pred = predictions[predictions[:,4] >= center_thresh]
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
        return mean_aps


class segmeanIOU(Metric):
    def get_score(self, batch, output, iter_id):
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