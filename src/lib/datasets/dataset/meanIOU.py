# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
import pdb
import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize

from detectron2.evaluation.evaluator import DatasetEvaluator
from mask_rcnn_stream import construct_semseg
from label_mappings import coco_class_groups, get_remap, combined_label_names
from detectron2.utils.visualizer import ColorMode, Visualizer

# basically taken from sem_seg_evaluation.py edited for BDD use
class BDDEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation going from COCO to BDD dataset
    """

    def __init__(self, cfg, num_classes, class_group_bdd, ignore_label=80, output_dir=None):
        """
        Args:
            num_classes (int): number of classes
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
                                                                                                                                                                                                                                                                                                     j,mcorresponding pixels should be ignored.
            output_dir (str): an output directory to dump results.
        """
        def convert_class_group_to_label(class_group):
            class_remap = {}
            for g in range(len(class_group)):
                for c in class_group[g]:
                    class_remap[c] = g
            return class_remap
        
        self._output_dir = output_dir
        self._num_classes = num_classes
        self._ignore_label = ignore_label
        self._N = num_classes + 1

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._class_map_bdd = convert_class_group_to_label(class_group_bdd)
        self._class_names = combined_label_names
        self._cfg = cfg
        # self._class_map_coco = convert_class_group_to_label(class_group_coco)

    def reset(self):
        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)
        self._predictions = []

    def convert_bdd_to_class_labels(self, inputs):
        # write some code that will convert
        gt = inputs['sem_seg']
        new_gt = np.full(gt.shape, self._num_classes)
        for k, v in self._class_map_bdd.items():
            new_gt[gt == k] = v
        return new_gt

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        coco_remap = get_remap(coco_class_groups)
        for i, (input, output) in enumerate(zip(inputs, outputs)):
            metadata = MetadataCatalog.get("coco_2017_test")
            # viz = Visualizer(np.array(inputs[0]['image']).transpose(1, 2, 0), metadata)
            # vis_output = viz.draw_instance_predictions(predictions=output['instances'].to(torch.device("cpu")))
            # cv2.imwrite(f"results/{i}.png", vis_output.get_image())
            # pdb.set_trace()
            output = construct_semseg(output, self._num_classes, coco_remap)
            pred = np.array(output, dtype=np.int)
            gt = self.convert_bdd_to_class_labels(input)
            

            # add to conf matrix for each image
            self._conf_matrix += np.bincount(
                self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N ** 2
            ).reshape(self._N, self._N)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """

        acc = np.full(self._N, np.nan, dtype=np.float)
        iou = np.full(self._N, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal().astype(np.float)
        pos_gt = np.sum(self._conf_matrix, axis=0).astype(np.float)
        # pdb.set_trace()
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix, axis=1).astype(np.float)
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
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
