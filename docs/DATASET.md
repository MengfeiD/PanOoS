# Prepare Datasets for Training POS

## General Overview

In order to train a model on known data from scratch, we use the same setup as followed by Mask2Former. A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

MaskFormer has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  cityscapes/
  mapillary_vistas/
  coco/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.


## Expected dataset structure for [cityscapes](https://www.cityscapes-dataset.com/downloads/):
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Note: to create labelTrainIds.png, first prepare the above structure, then run cityscapesescript with:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```

## Expected dataset structure for [Mapillary Vistas](https://www.mapillary.com/dataset/vistas):
```
mapillary_vistas/
  training/
    images/
    labels/
  validation/
    images/
    labels/
```

No preprocessing is needed for Mapillary Vistas on semantic and panoptic segmentation.


## Expected dataset structure for [COCO](https://cocodataset.org/#download):

```
coco/
  annotations/
    instances_train2017.json
    ood_seg_train2017/
  train2017/  # image files that are mentioned in the corresponding json
```


## Preparing `ood_seg_train2017` for Outlier Exposure Fine-Tuning

Please following [Meta-OoD](https://github.com/robin-chan/meta-ood/tree/master/preparation) and use their script to generate the binary segmentation masks for potential OoD objects from COCO dataset to be pasted on the driving scenes. The chosen objects are such that they do not semantically overlap with the known classes in Cityscapes.
Alternatively, the preprocessed `ood_seg_train2017` dataset can be downloaded from [here](https://drive.google.com/file/d/1IcyzU75XEDXqARmuAZ4G1xvlljAbiDoC/view?usp=sharing).

# Prepare OoD Segmentation Datasets for Evaluation

Download the evaluation datasets and organize them according to the following directory structures.

## [DenseOoS](https://drive.google.com/file/d/1EUXbsd_PqtWZlTbwZehUzpzffAo1D9cB/view?usp=sharing)
  ```
  DenseOoS/
    gtFine/
      val/
    leftImg8bit/
      val/
  ```

## [QuadOoS](https://drive.google.com/file/d/1Ip7TvDHT1OqNsh-yl6MY6FQEpnT6QJck/view?usp=sharing)
  ```
  QuadOoS/
    images/
    masks/
  ```

## [Road Anomaly](https://www.epfl.ch/labs/cvlab/data/road-anomaly/)

  In order to use the custom torch dataset defined in `datasets/road_anomaly.py` the dataset is assumed to follow the following structure:

  ```
  RoadAnomaly/
    RoadAnomaly_jpg/
      frame_list.json
      frames/
        image_name/
          labels_semantic.png
        image_name.jpg  # `image_name` here is a placeholder for any image name
        ...
      
  ```

  The custom dataset is used for evaluation and the root containing the `RoadAnomaly/` folder can be passed during evaluation to the script.
