import functools
import glob
import json
import logging
import multiprocessing as mp
import numpy as np
import os
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


def load_jacquard_instances(image_dir, to_polygons=True):
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
    for cat_id, subdir in enumerate(os.listdir(image_dir)):
        dir_ = os.path.join(image_dir, subdir)
        for grasps_file in glob.glob(f"{dir_}/*_grasps.txt"):
            assert os.path.isfile(grasps_file), grasps_file

            image_file = grasps_file.replace("_grasps.txt", "_RGB.png")
            assert os.path.isfile(image_file), image_file

            mask_file = grasps_file.replace("_grasps.txt", "_mask.png")
            assert os.path.isfile(mask_file), mask_file

            files.append((cat_id, image_file, mask_file, grasps_file))
    assert len(files), "No images found in {}".format(image_dir)

    logger = logging.getLogger(__name__)
    logger.info("Preprocessing jacquard annotations ...")
    # This is still not fast: all workers will execute duplicate works and will
    # take up to 10m on a 8GPU server.
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))

    ret = pool.map(
        functools.partial(jacquard_files_to_dict, to_polygons=to_polygons),
        files,
    )
    logger.info("Loaded {} images from {}".format(len(ret), image_dir))

    # Map ids to contiguous ids
    #dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(os.listdir(image_dir))}
    #for dict_per_image in ret:
    #    for anno in dict_per_image["annotations"]:
    #        anno["category_id"] = dataset_id_to_contiguous_id[anno["category_id"]]

    return ret


def jacquard_files_to_dict(files, to_polygons):
    """
    Parse jacquard annotation files to a dict.

    Args:
        files (tuple): consists of (image_file, instance_id_file, label_id_file, json_file)
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        A dict in Detectron2 Dataset format.
    """
    cat_id, image_file, mask_file, grasps_file = files

    annos = []

    # See also the official annotation parsing scripts at
    # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/instances2dict.py  # noqa
    with PathManager.open(mask_file, "rb") as f:
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
    anno["category_id"] = cat_id
    anno["iscrowd"] = False #True # TODO: add together with seg mask
    anno["bbox_mode"] = BoxMode.XYWHA_ABS
    with open(grasps_file) as f:
        for i, line in enumerate(f):
            xc, yc, a, w, h = [float(v) for v in line[:-1].split(';')]
            anno["bbox"] = (xc, yc, w, h, -a)
            annos.append(anno.copy())
            if i >= 3:
                break

    ret["annotations"] = annos
    return ret


def register_jacquard_instances(name, image_dir):
    DatasetCatalog.register(
        name,
        lambda x=image_dir: load_jacquard_instances(x, to_polygons=True),
    )
    MetadataCatalog.get(name).set(
        thing_classes=os.listdir(image_dir),
        image_dir=image_dir,
        evaluator_type="jacquard"
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
    Test the jacquard dataset loader.

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

    dirname = "jacquard-data-vis"
    os.makedirs(dirname, exist_ok=True)

    if args.type == "instance":
        dicts = load_jacquard_instances(
            args.image_dir, to_polygons=True
        )
        logger.info("Done loading {} samples.".format(len(dicts)))
        meta = Metadata().set(thing_classes=os.listdir(args.image_dir))

    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        visualizer = Visualizer(img, metadata=meta)
        vis = visualizer.draw_dataset_dict(d)
        # cv2.imshow("a", vis.get_image()[:, :, ::-1])
        # cv2.waitKey()
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)
