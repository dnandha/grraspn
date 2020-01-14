import functools
import glob
import json
import logging
import multiprocessing as mp
import numpy as np
import os
import re
from itertools import chain
import pycocotools.mask as mask_util
from PIL import Image

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.utils.comm import get_world_size
from fvcore.common.file_io import PathManager

try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    pass


class Grasp(object):
    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def a_rad(self):
        """
        :return: Angle of the grasp to the horizontal.
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi/2) % np.pi - np.pi/2

    @property
    def a(self):
        return 180/np.pi * self.a_rad

    @property
    def c(self):
        """
        :return: Rectangle center point
        """
        return self.points.mean(axis=0).astype(np.int)

    @property
    def x(self):
        return self.c[1]

    @property
    def y(self):
        return self.c[0]

    @property
    def w(self):
        """
        :return: Rectangle width (i.e. perpendicular to the axis of the grasp)
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def h(self):
        """
        :return: Rectangle height (i.e. along the axis of the grasp)
        """
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @staticmethod
    def load_grasps(f):
        def text_to_num(l, offset=(0,0)):
            x, y = l.split()
            return [int(round(float(y))) - offset[0],
                    int(round(float(x))) - offset[1]]

        while True:
         # Load 4 lines at a time, corners of bounding box.
             try:
                 p0 = f.readline()
                 if not p0:
                     break  # EOF
                 p1, p2, p3 = f.readline(), f.readline(), f.readline()
                 gr = np.array([
                     text_to_num(p0),
                     text_to_num(p1),
                     text_to_num(p2),
                     text_to_num(p3)
                 ])

                 yield Grasp(gr)

             except ValueError:
                 # Some files contain weird values.
                 continue

    @staticmethod
    def load_grasps_plain(f):
        for grasp in Grasp.load_grasps(f):
            yield (grasp.x, grasp.y, grasp.w, grasp.h, grasp.a)


def load_cornell_instances(image_dir, to_polygons=True):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    files = []
    for grasps_file in glob.glob(os.path.join(image_dir, "*cpos.txt")):
        assert os.path.isfile(grasps_file), grasps_file

        cat_id = int(re.search("pcd(\d+)cpos.txt", grasps_file).group(1))

        image_file = grasps_file.replace("cpos.txt", "r.png")
        assert os.path.isfile(image_file), image_file

        neg_grasps_file = grasps_file.replace("cpos.txt", "cneg.txt") 
        assert os.path.isfile(neg_grasps_file), neg_grasps_file

        files.append((cat_id, image_file, grasps_file, neg_grasps_file))
    assert len(files), "No images found in {}".format(image_dir)

    logger = logging.getLogger(__name__)
    logger.info("Preprocessing cornell annotations ...")
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))

    ret = pool.map(
        functools.partial(cornell_files_to_dict, to_polygons=to_polygons),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))

    # Map ids to contiguous ids
    #dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(os.listdir(image_dir))}
    #for dict_per_image in ret:
    #    for anno in dict_per_image["annotations"]:
    #        anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]

    return ret


def cornell_files_to_dict(files, to_polygons):
    """
    Parse cornell annotation files to a dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        A dict in Detectron2 Dataset format.
    """
    cat_id, image_file, grasps_file, neg_grasps_file = files

    annos = []

    # See also the official annotation parsing scripts at
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/instances2dict.py  # noqa
    with PathManager.open(image_file, "rb") as f:
        inst_image = np.asarray(Image.open(f), order="F")
    #flattened_ids = np.unique(inst_image)
    flattened_ids = [0, 255]

    ret = {
        "file_name": image_file,
        "image_id": os.path.basename(image_file),
        "height": inst_image.shape[0],
        "width": inst_image.shape[1],
    }

    #for instance_id in flattened_ids:
        #anno["iscrowd"] = False
        #mask = np.asarray(inst_image == instance_id, dtype=np.uint8, order="F")
        #anno["segmentation"] = mask

        #inds = np.nonzero(mask)
        #ymin, ymax = inds[0].min(), inds[0].max()
        #xmin, xmax = inds[1].min(), inds[1].max()
        #anno["bbox"] = (xmin, ymin, xmax, ymax)
        #if xmax <= xmin or ymax <= ymin:
        #    continue
        #anno["bbox_mode"] = BoxMode.XYXY_ABS


    # treat each grasp as an instance
    anno = {}
    anno["category_id"] = 0 # cat_id # TODO: assertion error
    anno["iscrowd"] = False #True # TODO: add together with seg mask
    anno["bbox_mode"] = BoxMode.XYWHA_ABS
    with open(grasps_file) as f:
        for xc, yc, w, h, a in Grasp.load_grasps_plain(f):
            # careful: potential mistake in cornell format description on website, jaw and opening interchanged!
            #print(xc, yc, opening, jaw, a)
            assert xc >= 0, f"neg x value {grasps_file}"
            assert yc >= 0, f"neg y value {grasps_file}"
            #assert a >= 0, f"neg a value {grasps_file}"
            assert w > 0, f"neg jaw value {grasps_file}"
            assert h > 0, f"neg opening value {grasps_file}"
            assert w*h >= 1, f"box area too small {grasps_file}"
            anno["bbox"] = (xc, yc, w, h, a)
            annos.append(anno.copy())

    ret["annotations"] = annos
    return ret


def register_cornell(name, image_dir):
    DatasetCatalog.register(
        name,
        lambda x=image_dir: load_cornell_instances(x, to_polygons=True),
    )
    MetadataCatalog.get(name).set(
        #thing_classes=os.listdir(image_dir), # TODO: add together with segmentation
        thing_classes=["grasp", "nograsp"],
        image_dir=image_dir,
        evaluator_type="cornell"
    )

    #sem_key = key.format(task="sem_seg")
    #DatasetCatalog.register(
    #    sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
    #)
    #MetadataCatalog.get(sem_key).set(
    #    image_dir=image_dir, gt_dir=gt_dir, evaluator_type="sem_seg", **meta
    #)


if __name__ == "__main__":
    """
    Test the cornell dataset loader.

    Usage:
        python -m detectron2.data.datasets.cityscapes \
            cityscapes/leftImg8bit/train cityscapes/gtFine/train
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("--type", choices=["instance", "semantic"], default="instance")
    args = parser.parse_args()
    from detectron2.data.catalog import Metadata
    from detectron2.utils.visualizer import Visualizer

    logger = setup_logger(name=__name__)

    dirname = "cornell-data-vis"
    os.makedirs(dirname, exist_ok=True)

    if args.type == "instance":
        dicts = load_cornell_instances(
            args.image_dir, to_polygons=True
        )
        logger.info("Done loading {} samples.".format(len(dicts)))
        meta = Metadata().set(thing_classes="thing")

    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        # cv2.imshow("a", vis.get_image()[:, :, ::-1])
        # cv2.waitKey()
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
