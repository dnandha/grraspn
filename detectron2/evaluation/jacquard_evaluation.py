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
        ANGLEMAX = 30

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

        mAP, mPrec, mRec, mAcc = 0, 0, 0, 0
        nTotal = len(self._predictions)
        mTps, mFps = 0,0
        for pred in self._predictions:
            file_name, scores, boxes, classes = pred
            boxes_gt = RotatedBoxes(list(load_grasps(file_name)))

            # init true positives, false positives
            tps, fps = [], []
            # sort by confidence/score
            boxes = boxes[np.argsort(-scores, kind='mergesort')]
            TOP_N = 1
            for j in range(TOP_N):
                box = boxes[j]
                angle = box.tensor.squeeze()[2]
                sector = classes[j]
                ovmax = float('-inf')
                for k in range(len(boxes_gt)):
                    box_gt = boxes_gt[k]
                    angle_gt = box_gt.tensor.squeeze()[2]
                    print(sector*10, angle_gt)
                    
                    # compute iou on GPU
                    iou = pairwise_iou_rotated(box, box_gt) # TODO: assumes len(gts)>len(scores)
                    # get best match
                    max_iou = torch.max(iou)
                    ovmax = max((ovmax, max_iou))
                if ovmax > OVTHRES and abs(angle-angle_gt) <= ANGLEMAX:
                    tps.append(1)
                    fps.append(0)
                    mTps += 1
                else:
                    fps.append(1)
                    tps.append(0)
                    mFps += 1

            # compute precision and recall
            fp = np.cumsum(np.array(fps))
            tp = np.cumsum(np.array(tps))
            rec = tp / np.maximum(TOP_N, torch.finfo(torch.float64).eps) # avoid divide by zero
            #brec = tp / np.maximum(len(boxes_gt), torch.finfo(torch.float64).eps) # avoid divide by zero
            prec = tp / np.maximum(tp + fp, torch.finfo(torch.float64).eps) # avoid divide by zero

            # let pascal voc compute ap
            ap = voc_ap(rec, prec)

            mAP += ap / nTotal

        acc = mTps / (mTps+mFps)
        ret = OrderedDict()
        ret["grasp"] = {"mAP": mAP*100, "mAcc:": acc*100}
        # TODO: add segm

        return ret
