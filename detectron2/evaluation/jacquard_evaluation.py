# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import glob
import logging
import os
import tempfile
from collections import OrderedDict
import torch
import numpy as np
from PIL import Image

from detectron2.data import MetadataCatalog
from detectron2.utils import comm

from .evaluator import DatasetEvaluator

from detectron2.structures import RotatedBoxes, pairwise_iou_rotated
from detectron2.evaluation.pascal_voc_evaluation import voc_ap 

class JacquardEvaluator(DatasetEvaluator):
    """
    Evaluate instance segmentation results using jacquard

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._predictions = []

    def reset(self):
        self._predictions.clear()

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            file_name = input["file_name"].replace("_RGB.png", "_grasps.txt") # TODO: use image_id instead
            instances = output["instances"].to(self._cpu_device)
            #proposals = output["proposals"].to(self._cpu_device)

            scores = instances.scores
            boxes = instances.pred_boxes
            classes = instances.pred_classes

            self._predictions.append((file_name, scores, boxes, classes))
            #self._predictions["proposals"].append(proposals)

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP" and "AP50".
        """
        OVTHRES = 0.25 # TODO: make this configurable

        def load_grasps(path): # TODO: duplicate code, see dataloader
            with open(path) as f:
                for i, line in enumerate(f):
                    # careful: potential mistake in jacquard format description on website, jaw and opening interchanged!
                    xc, yc, a, jaw, opening = [float(v) for v in line[:-1].split(';')]
                    # jaw = h, opening = w according to jacquard paper
                    yield (xc, yc, opening, jaw, -a)

        comm.synchronize()
        if not comm.is_main_process():
            return

        aps = []
        for pred in self._predictions:
            file_name, scores, boxes, classes = pred
            boxes_gt = RotatedBoxes(list(load_grasps(file_name)))

            tps, fps = [], []
            # TODO: sort by confidence score?
            #best_score_idx = torch.argmax(scores).item()
            #best_box = boxes[best_score_idx]
            for j in range(len(boxes)):
                box = boxes[j]
                ovmax = float('-inf')
                for k in range(len(boxes_gt)):
                    gt_box = boxes_gt[k]
                    iou = pairwise_iou_rotated(box, gt_box) # TODO: assumes len(gts)>len(scores)
                    max_iou = torch.max(iou)
                    ovmax = np.max((ovmax, max_iou))
                if ovmax > OVTHRES:
                    tps.append(1)
                    fps.append(0)
                else:
                    fps.append(1)
                    tps.append(0)

            # compute precision recall
            fp = np.cumsum(np.array(fps))
            tp = np.cumsum(np.array(tps))
            rec = tp / float(len(boxes))
            # avoid divide by zero in case the first detection matches a difficult gt
            prec = tp / np.maximum(tp + fp, torch.finfo(torch.float64).eps)
            ap = voc_ap(rec, prec)
            aps.append(ap)

        ret = OrderedDict()
        ret["grasp"] = {"mAP": np.mean(aps)*100}
        # TODO: add segm

        return ret
