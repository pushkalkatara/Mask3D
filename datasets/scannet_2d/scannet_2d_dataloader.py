import copy
import logging
import random
import numpy as np
import pycocotools.mask as mask_util
import itertools
from imageio import imread
from typing import List, Union
import torch
from torch.nn import functional as F
from operator import itemgetter
from natsort import natsorted
import cv2

from detectron2.data import DatasetCatalog
from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    polygons_to_bitmask
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import matplotlib.pyplot as plt

import ipdb
st = ipdb.set_trace

__all__ = ["ScannetDatasetMapper"]


def inpaint_depth(depth):
    """
    inpaints depth using opencv
    Input: torch tensor with depthvalues: H, W
    Output: torch tensor with depthvalues: H, W
    """
    depth_inpaint = cv2.inpaint(depth, (depth == 0).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    depth[depth == 0] = depth_inpaint[depth == 0]
    return depth

def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def _get_dummy_anno(num_classes):
    return {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if is_train:
        image_size = cfg.INPUT.IMAGE_SIZE
        min_scale = cfg.INPUT.MIN_SCALE
        max_scale = cfg.INPUT.MAX_SCALE

        augmentation = []
        depth_augmentation = None

        if cfg.INPUT.RANDOM_FLIP != "none":
            assert cfg.MODEL.DECODER_3D == False, "Random flip is not supported for 3D"
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                )
            )

        augmentation.extend([
            T.ResizeScale(
                min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
            ),
            T.FixedSizeCrop(crop_size=(image_size, image_size)),
        ])
    else:
        augmentation = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST_SAMPLING
            ),
        ]
        depth_augmentation = None

    return augmentation, depth_augmentation


class ScannetDatasetMapper:
    def __init__(
        self,
        is_train: bool,
        dataset_name: str,
        image_format: str,
        decoder_3d: bool = False,
        inpaint_depth: bool = False,
        use_instance_mask: bool = False,
        frame_left: int = 2,
        frame_right: int = 2,
        num_classes: int = 18
    ):
        # fmt: off
        self.is_train               = is_train
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.frame_right            = frame_right
        self.frame_left             = frame_left
        self.num_classes            = num_classes
        self.decoder_3d             = decoder_3d
        self.inpaint_depth          = inpaint_depth
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"

        self.context_dataset_dicts = DatasetCatalog.get(dataset_name)

    def get_current_image_id(self, dataset_dict, context):
        idx = None
        for i, context_img in enumerate(context):
            if dataset_dict['file_name'] == context_img['file_name']:
                return i
        return idx

    def convert_to_video_dict(self, dataset_dict, context):
        record = {}
        record["height"] = dataset_dict["height"]
        record["width"] = dataset_dict["width"]
        record["file_names"] = []
        record["valid_file_names"] = []
        record["segment_file_names"] = []
        record["depth_file_names"] = []
        record["pose_file_names"] = []
        record["color_intrinsics_files"] = []
        record["depth_intrinsics_files"] = []
        record["annotations"] = []

        # check why its not sorted from register dataset itself.
        context = natsorted(context, key=itemgetter(*['file_name']))

        if self.frame_left > 0:
            idx = self.get_current_image_id(dataset_dict, context)
            left_ids = [(idx - i - 1) % len(context) for i in range(self.frame_left)]

            if len(left_ids) == 1:
                left_contexts = [itemgetter(*left_ids)(context)]
            else:
                left_contexts = list(itemgetter(*left_ids)(context))
                left_contexts.reverse()

            for lc in left_contexts:
                record["file_names"].append(lc['file_name'])
                record["valid_file_names"].append(lc['valid_file'])
                record["segment_file_names"].append(lc['segment_file'])
                record["depth_file_names"].append(lc['depth_file'])
                record["pose_file_names"].append(lc['pose_file'])
                record["color_intrinsics_files"].append(lc['color_intrinsics_file'])
                record["depth_intrinsics_files"].append(lc['depth_intrinsics_file'])
                record["annotations"].append(lc['annotations'])

        record["file_names"].append(dataset_dict['file_name'])
        record["valid_file_names"].append(dataset_dict['valid_file'])
        record["segment_file_names"].append(dataset_dict['segment_file'])
        record["depth_file_names"].append(dataset_dict['depth_file'])
        record["pose_file_names"].append(dataset_dict['pose_file'])
        record["color_intrinsics_files"].append(dataset_dict['color_intrinsics_file'])
        record["depth_intrinsics_files"].append(dataset_dict['depth_intrinsics_file'])
        record["annotations"].append(dataset_dict['annotations'])   

        if self.frame_right > 0:
            right_ids = [(idx + i + 1) % len(context) for i in range(self.frame_right)]
            if len(right_ids) == 1:
                right_contexts = [itemgetter(*right_ids)(context)]
            else:
                right_contexts = list(itemgetter(*right_ids)(context))

            for rc in right_contexts:
                record["file_names"].append(rc['file_name'])
                record["valid_file_names"].append(rc['valid_file'])
                record["segment_file_names"].append(rc['segment_file'])
                record["depth_file_names"].append(rc['depth_file'])
                record["pose_file_names"].append(rc['pose_file'])
                record["color_intrinsics_files"].append(rc['color_intrinsics_file'])
                record["depth_intrinsics_files"].append(rc['depth_intrinsics_file'])
                record["annotations"].append(rc['annotations'])

        record["length"] = len(record["file_names"])
        record["image_id"] = dataset_dict['image_id']

        return record

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video
        
        Returns:
            dict: a format that builtin models in ...
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        context = list(itemgetter(*dataset_dict['context_ids'])(self.context_dataset_dicts))
        dataset_dict = self.convert_to_video_dict(dataset_dict, context)

        video_length = dataset_dict["length"]
        selected_idx = range(video_length)
        eval_idx = video_length // 2

        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)
        if self.decoder_3d:
            depth_file_names = dataset_dict.pop("depth_file_names", None)
            if self.inpaint_depth:
                # replace "depth" in "dpeth_file_names" path with "depth_inpainted" in dataset_dict
                depth_file_names = [
                    depth_file_names[i].replace("depth", "depth_inpainted") for i in range(len(depth_file_names))
                ]
            color_intrinsics_files = dataset_dict.pop("color_intrinsics_files", None)
            depth_intrinsics_files = dataset_dict.pop("depth_intrinsics_files", None)
            pose_file_names = dataset_dict.pop("pose_file_names", None)
            valid_file_names = dataset_dict.pop("valid_file_names", None)
            segment_file_names = dataset_dict.pop("segment_file_names", None)
        
        dataset_dict["images"] = []
        dataset_dict["padding_masks"] = []
        dataset_dict["image"] = None
        dataset_dict["instances"] = None
        dataset_dict["instances_all"] = []
        dataset_dict["file_names"] = []
        dataset_dict["file_name"] = None

        if self.decoder_3d:
            dataset_dict["depths"] = []
            dataset_dict["poses"] = []
            dataset_dict["depth_intrinsics"] = []
            dataset_dict["color_intrinsics"] = []
            dataset_dict["valids"] = []
            dataset_dict["segments"] = []

            dataset_dict["depth_file_names"] = []
            dataset_dict["pose_file_names"] = []
            dataset_dict["color_intrinsics_files"] = []
            dataset_dict["depth_intrinsics_files"] = []
            dataset_dict["valid_file_names"] = []
            dataset_dict["segment_file_names"] = []

        for frame_idx in selected_idx:
            if eval_idx == frame_idx:
                dataset_dict["file_name"] = file_names[frame_idx]
            dataset_dict["file_names"].append(file_names[frame_idx])

            if self.decoder_3d:
                dataset_dict["valid_file_names"].append(valid_file_names[frame_idx])
                dataset_dict["segment_file_names"].append(segment_file_names[frame_idx])
                dataset_dict["depth_file_names"].append(depth_file_names[frame_idx])
                dataset_dict["pose_file_names"].append(pose_file_names[frame_idx])
                dataset_dict["color_intrinsics_files"].append(color_intrinsics_files[frame_idx])
                dataset_dict["depth_intrinsics_files"].append(depth_intrinsics_files[frame_idx])


            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)
            padding_mask = np.ones(image.shape[:2])

            aug_input = T.AugInput(image)
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            image = aug_input.image
            # utils.check_image_size(dataset_dict, image)

            # get padding mask for segmentation
            padding_mask = transforms.apply_segmentation(padding_mask)
            padding_mask = ~ padding_mask.astype(bool)

            if self.decoder_3d:
                # check depth scale / 000.0 
                depth = imread(depth_file_names[frame_idx]).astype(np.float32)
                depth = depth / 1000.0
                # depth_aug_input = T.AugInput(depth)
                # depth_aug_input, depth_transforms = T.apply_transform_gens(self.depth_tfm_gems, depth_aug_input)
                # depth = depth_aug_input.image
                pose = np.loadtxt(pose_file_names[frame_idx])
                try:
                    valid = np.load(valid_file_names[frame_idx])
                    segment = np.load(segment_file_names[frame_idx])
                except:
                    valid = np.ones_like(depth)
                    # nor idea if we take ones like segments.
                    # check what scenes are going in this except
                    segment = np.ones_like(depth)

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]

            if len(annos):
                assert "segmentation" in annos[0]
            segms = [obj["segmentation"] for obj in annos]
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            if self.decoder_3d:
                depth = torch.as_tensor(np.ascontiguousarray(depth))
                pose = torch.from_numpy(pose).float()
                valid = torch.from_numpy(valid).bool()
                segment = torch.from_numpy(segment)
            masks = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]
            classes = [int(obj["category_id"]) for obj in annos]
            classes = torch.tensor(classes, dtype=torch.int64)
            instance_ids = [int(obj["semantic_instance_id_scannet"]) for obj in annos]
            instance_ids = torch.tensor(instance_ids, dtype=torch.int64)


            image_shape = (image.shape[-2], image.shape[-1])
            if self.decoder_3d:
                depth_shape = (depth.shape[-2], depth.shape[-1])
                assert depth_shape == image_shape
        
            dataset_dict["images"].append(image)
            dataset_dict["padding_masks"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
            if self.decoder_3d:
                dataset_dict["depths"].append(depth)
                dataset_dict["poses"].append(pose)
                dataset_dict["valids"].append(valid)
                dataset_dict["segments"].append(segment)

            if frame_idx == eval_idx:
                dataset_dict["image"] = image

            instances = Instances(image_shape)
            instances.gt_classes = classes
            instances.instance_ids = instance_ids
            if len(masks) == 0:
                instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
            else:
                masks = BitMasks(torch.stack(masks))
                instances.gt_masks = masks.tensor
            
            dataset_dict["instances_all"].append(instances)
            if frame_idx == eval_idx:
                dataset_dict["instances"] = instances
        
        return dataset_dict