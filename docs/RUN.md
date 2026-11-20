## Training

In order to replicate any of our experiments, we provide the config files for all the models in the [Model Zoo](MODEL_ZOO.md). Given the config file, a training experiment can be run using the following command:

```bash
python train_net.py \
--config-file PATH_TO_CONFIG_FILE \
--num-gpus    NUM_GPUS \
OUTPUT_DIR    PATH_TO_STORE_CKPTS_AND_LOGS
```

For example, to train a CLIP (ConvNeXt-Large) with 1 decoder layer, on 4 gpus and store the outputs in a folder named `model_logs/clip_1dl` you can use the following:

```bash
python train_net.py \
--config-file configs/cityscapes/semantic-segmentation/clip/clip_convnext_large_eval_ade20k_1dl.yaml \
--num-gpus 4 \
OUTPUT_DIR  model_logs/clip_1dl/
```
or
```bash
python train_net.py \
--config-file ckpts/clip_1dl/config.yaml \
--num-gpus 4 \
OUTPUT_DIR  model_logs/clip_1dl/
```

## Outlier Exposure Fine-tuning
The models are trained on cityscapes dataset only and are used as a starting point for finetuning with outlier supervision `model_logs/clip_1dl_rba_ood_map_coco`.

```bash
python train_net.py \
--config-file configs/mapillary-vistas/semantic-segmentation/finetuning_with_cityscapes/clip_convnext_large_eval_ade20k_1dl_coco_mix_finetune_mapillary.yaml \
--num-gpus 4 \
OUTPUT_DIR  model_logs/clip_1dl_rba_ood_map_coco/
```
or
```bash
python train_net.py \
--config-file ckpts/clip_1dl_rba_ood_map_coco/config.yaml \
--num-gpus 4 \
OUTPUT_DIR  model_logs/clip_1dl_rba_ood_map_coco/
```

## Evaluation

A simple usage for the script is as follows:

```bash
python evaluate_ood.py \
--out_path results/ \
--models_folder model_logs/ \
--datasets_folder PATH_TO_DATASETS_ROOT
```

Notes:

* The OoD datasets are setup as described in [Datasets Prepration](docs/DATASET.md)
* The parameter `--models_folder` is a path to a folder that contains multiple folders, where each folder corresponds to a model. In a model's folder the scripts expects to files: `config.yaml` and its checkpoint `model_final.pth`.

The scripts supports more finegrained options like selecting subsets of the models in a folder or the datasets. Please check `evaluate_ood.py` for descriptions of the options.