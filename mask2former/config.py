# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # NOTE: Custom stuff added by Nazir for convenience

    # OOD Training
    cfg.INPUT.OOD_LABEL = 254
    cfg.MODEL.MASK_FORMER.OUTLIER_SUPERVISION = False
    cfg.MODEL.MASK_FORMER.OUTLIER_LOSS_TARGET = "none"
    cfg.MODEL.MASK_FORMER.INLIER_UPPER_THRESHOLD = -1.0
    cfg.MODEL.MASK_FORMER.OUTLIER_LOWER_THRESHOLD = 5.0   # OUTLIER_LOWER_THRESHOLD: 5.0, alpha for RbA loss function
    cfg.MODEL.MASK_FORMER.OUTLIER_WEIGHT = 1.0

    cfg.INPUT.OOD_PROB = 0.1
    cfg.INPUT.COCO_ROOT = "coco/"
    cfg.INPUT.COCO_PROXY_SIZE = 300

    # BPDL fintuning
    cfg.MODEL.MASK_FORMER.PDL_LOSS = False
    cfg.MODEL.MASK_FORMER.BPDL_LOSS = False
    cfg.MODEL.MASK_FORMER.BPDL_WEIGHT = 0.01
    cfg.MODEL.MASK_FORMER.BPDL_ALPHA = 0.1
    cfg.MODEL.MASK_FORMER.BPDL_D = 40
    cfg.MODEL.MASK_FORMER.BPDL_S = 15

    # Matcher
    cfg.MODEL.MASK_FORMER.MATCHER = 'HungarianMatcher'

    # Finetuning
    cfg.MODEL.FREEZE_BACKBONE = False
    cfg.MODEL.FREEZE_PIXEL_DECODER = False
    cfg.MODEL.FREEZE_TRANSFORMER_DECODER = False
    cfg.MODEL.FREEZE_TRANSFORMER_DECODER_EXCEPT_MLP = False # This takes precedence over FREEZE_TRANSFORMER_DECODER
    cfg.MODEL.FREEZE_TRANSFORMER_DECODER_EXCEPT_MLP_AND_OOD_PRED = False
    cfg.MODEL.FREEZE_TRANSFORMER_DECODER_EXCEPT_OBJECT_QUERIES = False # This takes precedence over FREEZE_TRANSFORMER_DECODER_EXCEPT_MLP

    # Loss Functions
    cfg.MODEL.MASK_FORMER.SMOOTHNESS_LOSS = False
    cfg.MODEL.MASK_FORMER.SMOOTHNESS_WEIGHT = 3e-6
    cfg.MODEL.MASK_FORMER.SPARSITY_LOSS = False
    cfg.MODEL.MASK_FORMER.SPARSITY_WEIGHT = 5e-4
    cfg.MODEL.MASK_FORMER.SMOOTHNESS_SCORE = "energy"
    cfg.MODEL.MASK_FORMER.GAMBLER_LOSS = False
    cfg.MODEL.MASK_FORMER.GAMBLER_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.PEBAL_REWARD = 4.5
    cfg.MODEL.MASK_FORMER.PEBAL_OOD_REG = 0.1 
    cfg.MODEL.MASK_FORMER.OUTLIER_LOSS_FUNC = "max"
    cfg.MODEL.MASK_FORMER.DENSE_HYBRID_LOSS = False
    cfg.MODEL.MASK_FORMER.DENSE_HYBRID_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DENSE_HYBRID_BETA = 0.03

    cfg.MODEL.MASK_FORMER.SCORE_NORM = "none"
    cfg.MODEL.MASK_FORMER.OUTLIER_LOSS_FUNC = "squared_hinge"

    # Input Format
    cfg.INPUT.REPEAT_INSTANCE_MASKS = 1

    # Ablations
    cfg.SOLVER.FORCE_REGION_PARTITION = False
    cfg.MODEL.MASK_FORMER.USE_POINT_REND = False
    # logger
    cfg.SOLVER.USE_WANDB = False
    cfg.SOLVER.WANDB_PROJECT = 'mask2former'
    cfg.SOLVER.WANDB_NAME = ''

    # Open Panoptic Segmentation
    cfg.MODEL.MASK_FORMER.UNLABELED_REGION = True
    cfg.MODEL.MASK_FORMER.SEM_SEG_UNLABELED_REGION = True
    cfg.MODEL.MASK_FORMER.OPEN_PANOPTIC = True
    cfg.DATASETS.UNSEEN_LABEL_SET = 'datasets/unknown/unknown_K20.txt'

def add_pos_config(cfg):
    # CLIP model config in POS
    cfg.MODEL.CLIP = CN()
    cfg.MODEL.CLIP.CLIP_MODEL_NAME = "convnext_large_d_320"
    cfg.MODEL.CLIP.CLIP_PRETRAINED_WEIGHTS = "laion2b_s29b_b131k_ft_soup"
    cfg.MODEL.CLIP.EMBED_DIM = 768
    cfg.MODEL.CLIP.GEOMETRIC_ENSEMBLE_ALPHA = 0.4
    cfg.MODEL.CLIP.GEOMETRIC_ENSEMBLE_BETA = 0.8
    cfg.MODEL.CLIP.ENSEMBLE_ON_VALID_MASK = False