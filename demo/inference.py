import fvcore
from predictor import VisualizationDemo
from detectron2.config import get_cfg
import os
import numpy as np
from io import BytesIO
from imageio import imread
import glob
import sys

## TODO: add argparser and make configurable

imgdir = sys.argv[1]
outfile = sys.argv[2]

def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file('configs/grasp_rcnn_R_50_FPN_3x.yaml')
    cfg.merge_from_list(['MODEL.WEIGHTS', 'output/model_final.pth'])
    # Set score_threshold for builtin models
    confidence_threshold = 0.5
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg

cfg = setup_cfg()
demo = VisualizationDemo(cfg)

with open(outfile, 'w') as f:
    for imgpath in glob.glob(os.path.join(imgdir, "*/*_RGB.png")):
        img = imread(imgpath, format='png')
        img = img[:,:,:3][:,:,::-1]
        predictions, visualized_output = demo.run_on_image(img)
        boxes = predictions['instances'].pred_boxes.tensor.cpu()
        if len(boxes) > 0:
            scores = predictions['instances'].scores.cpu()
            boxes = boxes[np.argsort(-scores, kind='mergesort')]
            # TODO: do detection here and return grasp
            p = [boxes[i].squeeze().tolist() for i in range(len(boxes))]
            imgname = os.path.basename(imgpath).split('_RGB')[0]
            x, y, opening, jaw, a = p[0]
            grasp = "{:.4f};{:.4f};{:.4f};{:.4f};{:.4f}".format(x,y,-a,opening,jaw)
            print(imgname, grasp)
            f.write(imgname+'\n')
            f.write(grasp+'\n')
