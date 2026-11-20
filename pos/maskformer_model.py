# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
import cv2 
import numpy as np
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher, FixedMatcher
import fvcore.nn.weight_init as weight_init

VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]

# Bilevel Prompt Distribution Learning
DISTRIBUTION_PROMPT = [
    "a photo of an {} item.",
    "This is a photo of an {} item",
    "There is an {} item in the urban or rural scenes",
    "There is the {} item in the urban or rural scenes",
    "a photo of an {} item in the urban or rural scenes",
    "a photo of a small {} item.",
    "a photo of a medium {} item.",
    "a photo of a large {} item.",
    "This is a photo of a small {} item.",
    "This is a photo of a medium {} item.",
    "This is a photo of a large {} item.",
    "There is a small {} item in the urban or rural scenes.",
    "There is a medium {} item in the urban or rural scenes.",
    "There is a large {} item in the urban or rural scenes.",
]

@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        force_region_partition: bool,
        outlier_supervision: bool,
        open_panoptic: bool,
        mask_dim: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        # metadata for training
        self.train_metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(
            pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(
            pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.force_region_partition = force_region_partition
        self.outlier_supervision = outlier_supervision
        self.open_panoptic = open_panoptic

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.train_text_classifier = None
        self.train_distribution_classifier = None
        self.void_embedding = nn.Embedding(1, backbone.dim_latent) # use this for void
        _, self.train_num_templates, self.train_class_names, self.train_distributions = self.prepare_prompts_from_metadata(self.metadata, self.train_metadata)

        self.text_proj = nn.Linear(backbone.dim_latent, mask_dim)
        weight_init.c2_xavier_fill(self.text_proj)

    def prepare_prompts_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # # this could be for insseg, where only thing_classes are available
            # class_names = split_labels(metadata.thing_classes)
            # train_class_names = split_labels(train_metadata.thing_classes)
            print("No stuff classes found in metadata, please check the stuff classes for Cityscapes 19 categories.")
        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names))
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)
        
        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)

        def fill_all_templates_distribution(x_=''):
            res = []
            for template in DISTRIBUTION_PROMPT:
                res.append(template.format(x_))
            return res

        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names

        templated_distribution = []
        for d in ["inlier", "outlier"]:
            templated_distri = fill_all_templates_distribution(d)
            templated_distribution += templated_distri
        distributions = templated_distribution
        
        return category_overlapping_mask, num_templates, class_names, distributions
    
    def get_text_classifier(self):
        if self.train_text_classifier is None:
            text_classifier = []
            distribution_classifier = []
            # this is needed to avoid oom, which may happen when num of class is large
            bs = 128
            for idx in range(0, len(self.train_class_names), bs):
                text_classifier.append(self.backbone.get_text_classifier(self.train_class_names[idx:idx+bs], self.device).detach())
            text_classifier = torch.cat(text_classifier, dim=0)
            
            text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)

            # inlier or outlier supervision
            for idx in range(0, len(self.train_distributions), bs):
                distribution_classifier.append(self.backbone.get_text_classifier(self.train_distributions[idx:idx+bs], self.device).detach())
            distribution_classifier = torch.cat(distribution_classifier, dim=0)

            distribution_classifier = distribution_classifier.reshape(distribution_classifier.shape[0]//len(DISTRIBUTION_PROMPT),
                                                                        len(DISTRIBUTION_PROMPT), distribution_classifier.shape[-1]).mean(1)
            
            self.train_text_classifier = text_classifier
            self.train_distribution_classifier = distribution_classifier
        return self.train_text_classifier, self.train_num_templates, self.train_distribution_classifier

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        smoothness_weight = cfg.MODEL.MASK_FORMER.SMOOTHNESS_WEIGHT
        sparsity_weight = cfg.MODEL.MASK_FORMER.SPARSITY_WEIGHT
        gambler_weight = cfg.MODEL.MASK_FORMER.GAMBLER_WEIGHT
        outlier_weight = cfg.MODEL.MASK_FORMER.OUTLIER_WEIGHT
        densehybrid_weight = cfg.MODEL.MASK_FORMER.DENSE_HYBRID_WEIGHT
        pdl_weight = bpdl_weight = cfg.MODEL.MASK_FORMER.BPDL_WEIGHT
        # building criterion
        matcher_type = cfg.MODEL.MASK_FORMER.MATCHER
        if matcher_type == "HungarianMatcher":
            matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_mask=mask_weight,
                cost_dice=dice_weight,
                num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            )
        elif matcher_type == "FixedMatcher":

            if sem_seg_head.num_classes != cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES:
                raise ValueError(
                    "When using FixedMatcher, number of object queries must be equal to number of classes")

            matcher = FixedMatcher()

        else:
            raise ValueError(f"Given Matcher ({matcher_type}) is not defined")

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
            "smoothness_loss": smoothness_weight,
            "sparsity_loss": sparsity_weight,
            "outlier_loss": outlier_weight,
            "gambler_loss": gambler_weight,
            "densehybrid_loss": densehybrid_weight,
            "pdl_loss": pdl_weight,
            "bpdl_loss": bpdl_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        if cfg.MODEL.MASK_FORMER.GAMBLER_LOSS:
            losses = ["gambler"]
        if cfg.MODEL.MASK_FORMER.DENSE_HYBRID_LOSS:
            losses = ["densehybrid"]
        if cfg.MODEL.MASK_FORMER.SMOOTHNESS_LOSS:
            losses.append("smoothness")
        if cfg.MODEL.MASK_FORMER.SPARSITY_LOSS:
            losses.append("sparsity")
        if cfg.MODEL.MASK_FORMER.OUTLIER_SUPERVISION:
            losses.append("outlier")
        if cfg.MODEL.MASK_FORMER.PDL_LOSS:
            losses.append("pdl")
        if cfg.MODEL.MASK_FORMER.BPDL_LOSS:
            losses.append("bpdl")
        
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            smoothness_score=cfg.MODEL.MASK_FORMER.SMOOTHNESS_SCORE,
            outlier_loss_target=cfg.MODEL.MASK_FORMER.OUTLIER_LOSS_TARGET,
            inlier_upper_threshold=cfg.MODEL.MASK_FORMER.INLIER_UPPER_THRESHOLD,
            outlier_lower_threshold=cfg.MODEL.MASK_FORMER.OUTLIER_LOWER_THRESHOLD,
            score_norm=cfg.MODEL.MASK_FORMER.SCORE_NORM,
            outlier_loss_func=cfg.MODEL.MASK_FORMER.OUTLIER_LOSS_FUNC,
            pebal_reward=cfg.MODEL.MASK_FORMER.PEBAL_REWARD,
            ood_reg=cfg.MODEL.MASK_FORMER.PEBAL_OOD_REG,
            densehybrid_beta=cfg.MODEL.MASK_FORMER.DENSE_HYBRID_BETA,
            bpdl_alpha=cfg.MODEL.MASK_FORMER.BPDL_ALPHA,
            bpdl_d=cfg.MODEL.MASK_FORMER.BPDL_D,
            bpdl_s=cfg.MODEL.MASK_FORMER.BPDL_S,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "force_region_partition": cfg.SOLVER.FORCE_REGION_PARTITION,
            "outlier_supervision": cfg.MODEL.MASK_FORMER.OUTLIER_SUPERVISION,
            "open_panoptic": cfg.MODEL.MASK_FORMER.OPEN_PANOPTIC,
            "mask_dim": cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, include_void=False, return_separately=False, 
                return_aux=False, return_ood_pred=False, panoptic_ood_threshold=-0.3, 
                panoptic_pixel_min=200, return_panoptic_ood=False):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        text_classifier, _, distribution_classifier = self.get_text_classifier()
        text_classifier = torch.cat([text_classifier, self.void_embedding.weight, distribution_classifier], dim=0)
        text_classifier = self.text_proj(text_classifier)
        features["text_embeddings"] = text_classifier
        
        outputs = self.sem_seg_head(features)

        if self.force_region_partition:
            B, N, H, W = outputs["pred_masks"].shape
            outputs["pred_masks"] = outputs["pred_masks"].softmax(dim=1)

        if self.training:
            # text classifier for Prompt Distribution Learning in training
            if self.outlier_supervision:
                outputs.update({"text_embeddings": text_classifier})

            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(
                    self.device) for x in batched_inputs]
                if self.outlier_supervision:
                    outlier_masks = [x["outlier_mask"].to(self.device) for x in batched_inputs]
                    sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
                    targets = self.prepare_targets(gt_instances, images, outlier_masks, sem_seg)
                else:
                    targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"] # B x N x K
            mask_pred_results = outputs["pred_masks"] # B x N x H x W
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            if return_aux:
                aux = outputs["aux_outputs"]

            if return_ood_pred:
                ood_pred = outputs["ood_pred"]
                ood_pred = F.interpolate(ood_pred, size=images.image_sizes[0], mode='bilinear', align_corners=True)
            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    if include_void:
                        r = retry_if_cuda_oom(self.semantic_inference_with_void)(
                            mask_cls_result, mask_pred_result)
                    else:
                        r = retry_if_cuda_oom(self.semantic_inference)(
                            mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(
                            r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(
                        mask_cls_result, mask_pred_result, self.open_panoptic, 
                        panoptic_ood_threshold, panoptic_pixel_min, return_panoptic_ood)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(
                        mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            if return_aux:
                return processed_results, aux
            if return_ood_pred:
                return processed_results, ood_pred

            if return_separately:
                return processed_results, mask_cls_result, mask_pred_result

            return processed_results

    def prepare_targets(self, targets, images, outlier_masks=None, sem_seg=None):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for i, targets_per_image in enumerate(targets):
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1],
                         : gt_masks.shape[2]] = gt_masks
            
            entry = {
                "labels": targets_per_image.gt_classes,
                "masks": padded_masks,
            }
            if outlier_masks is not None:
                entry["outlier_masks"] = outlier_masks[i]
            if sem_seg is not None:
                entry["sem_seg"] = sem_seg[i]
            new_targets.append(entry)
                
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        # K x H x W
        return semseg

    def semantic_inference_with_void(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred, open_panoptic=False, ood_threshold=-0.1, pixel_min=300, return_ood_pred=False):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (
            scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros(
            (h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(
                                pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(
                                pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
            
            if open_panoptic:
                
                mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
                semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
                ood_mask = -(semseg.tanh()).sum(0)

                ood_mask_binary = ood_mask > ood_threshold

                # compute connected components of ood_mask
                ood_mask_binary = ood_mask_binary.cpu().numpy().astype(np.uint8)
                ood_mask_binary = cv2.morphologyEx(ood_mask_binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                ood_mask_binary = cv2.morphologyEx(ood_mask_binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                num_labels, labels_im = cv2.connectedComponents(ood_mask_binary, connectivity=4)
                labels_im = torch.from_numpy(labels_im).to(self.device)

                for i in range(1, num_labels):
                    mask = (labels_im == i) & (panoptic_seg == 0)
                    if mask.sum() < pixel_min:
                        continue
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": True,
                            "category_id": 255,
                        }
                    )

                if return_ood_pred:
                    return panoptic_seg, segments_info, ood_mask

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(
            0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(
            0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(
            1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
