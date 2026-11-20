# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg


def _get_kitti360_cityscapes_meta():
    stuff_classes = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]
    assert len(stuff_classes) == 19, f"Given: {len(stuff_classes)}"

    stuff_colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    assert len(stuff_colors) == 19

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_all_kitti360_train(root):
    root = os.path.join(root, "APS_KITTI_360/Image")
    meta = _get_kitti360_cityscapes_meta()
    image_dir = os.path.join(root, "train")
    gt_dir = os.path.join(root, "train").replace("Image", "Semantic")
    name = f"kitti360_cityscapes_sem_seg_train"
    DatasetCatalog.register(
        name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
    )
    MetadataCatalog.get(name).set(
        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="sem_seg",
        ignore_label=255,  # different from other datasets, Mapillary Vistas sets ignore_label to 65
        **meta,
    )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_kitti360_train(_root)
