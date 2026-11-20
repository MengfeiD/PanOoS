# Model Zoo

## Introduction

The configs for the pretrained models are provided under `ckpts/` folder, under which there are multiple folders and each folder represents a separate model, when downloading the checkpoints you should place the `model_final.pth` in its corresponding model_folder. `ckpts/` would like something like this:
```
ckpts/
  clip_1dl/
    config.yaml
    model_final.pth
  clip_1dl_rba_ood_map_coco/
    config.yaml
    model_final.pth
  ...
```

## Pretrained Models for Initialization

It's common to initialize from backbone models pre-trained on ImageNet classification tasks. The following backbone models are available through the original [Mask2Former](https://github.com/facebookresearch/Mask2Former) repo:

* [R-50.pkl (torchvision)](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl): converted copy of [torchvision's ResNet-50](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet50) model.
  More details can be found in [the conversion script](tools/convert-torchvision-to-d2.py).
* [R-103.pkl](https://dl.fbaipublicfiles.com/detectron2/DeepLab/R-103.pkl): a ResNet-101 with its first 7x7 convolution replaced by 3 3x3 convolutions. This modification has been used in most semantic segmentation papers (a.k.a. ResNet101c in RbA). We pre-train this backbone on ImageNet using the default recipe of [pytorch examples](https://github.com/pytorch/examples/tree/master/imagenet).
* [Swin-{B-L}](tools/README.md): pretrained on ImageNet, details on how to download them models and transform them to detectron2 format can be found [here](tools/README.md).

**Note**: according the config files we provide, it is assumed that the pretrained checkpoints are placed under the `pretrained/` folder. Make sure they are either placed there or that the config file is overwritten with the desired path when running an experiment.
