import logging
import numpy as np
from natsort import natsorted
import operator
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances

logger = logging.getLogger(__name__)

__all__ = ["load_scannet_json", "register_scannet_instances"]


NAME_MAP = { 1: 'cabinet', 2: 'bed', 3: 'chair', 4: 'sofa', 5: 'table', 6: 'door', 7: 'window',
             8: 'bookshelf', 9: 'picture', 10: 'counter', 11: 'desk', 12: 'curtain', 13: 'refridgerator',
             14: 'shower curtain', 15: 'toilet', 16: 'sink', 17:' bathtub',  18: 'otherfurniture'}

SCANNET_CATEGORIES = [
    {'id': key, 'name': item, 'supercategory': 'nyu40' } for key, item in NAME_MAP.items() 
]

def _get_scannet_instances_meta():
    thing_ids = [k["id"] for k in SCANNET_CATEGORIES]
    assert len(thing_ids) == 18, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SCANNET_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def get_scenes(image_dataset):
    '''
    takes in a video dataset and retuns dict of scene to images mapping
    returns : dict{scene_id: str, image_dataset_ids: []}
    '''
    scenes = {}
    for idx, img in enumerate(image_dataset):
        scene_id = img['file_name'].split('/')[-3]
        if scene_id in scenes:
            scenes[scene_id].append(idx)
        else:
            scenes[scene_id] = [idx]
    return scenes

def make_video_from_frames(imgs):
    # sort images by filename
    imgs = natsorted(imgs, key=operator.itemgetter(*['file_name']))
    file_names = []
    annotations = []
    for img in imgs:
        file_names.append(img['file_name'])
        annotations.append(img['annotations'])
    return file_names, annotations

def get_context_records(idx, img, scenes):
    scene_id = img['file_name'].split('/')[-3]
    dataset_ids = scenes[scene_id]
    return dataset_ids
    
def aggregate_images_by_sceneid(image_dataset):
    '''
    Takes in all scannet images, and add its multiview images as contexts.
    '''
    scenes = get_scenes(image_dataset)
    
    # add depth poses for reprojection
    for idx, img in enumerate(image_dataset):
        # add depth, pose, intrinsics
        img['depth_file'] = img['file_name'].replace('color', 'depth').replace('jpg', 'png')
        img['depth_intrinsics_file'] = os.path.splitext(img['file_name'])[0].replace('color', 'intrinsics_depth.txt')
        img['color_intrinsics_file'] = os.path.splitext(img['file_name'])[0].replace('color', 'intrinsics_color.txt')
        img['pose_file'] = img['file_name'].replace('color', 'pose').replace('jpg', 'txt')
        img['valid_file'] = img['file_name'].replace('color', 'instance_newest').replace('jpg', 'npy')
        directory, filename = os.path.split(img['file_name'])
        file_tail = filename.split('.')[0]
        filename = filename.replace(f'{file_tail}', f'segment_{file_tail}')
        segment_file = os.path.join(directory, filename).replace('color', 'instance_newest').replace('jpg', 'npy')
        img['segment_file'] = segment_file
        #/projects/katefgroup/language_grounding/SEMSEG_100k/frames_square/scene0191_00/instance_newest/0.npy
        # add context image ids
        img['context_ids'] = get_context_records(idx, img, scenes)

    '''
    for scene_id, imgs in scenes.items():
        record = {}
        record["file_names"], record["annotations"] = make_video_from_frames(imgs)
        record["height"] = imgs[0]["height"]
        record["width"] = imgs[0]["width"]
        record["length"] = len(imgs)
        record["scene_id"] = scene_id
        dataset_dicts.append(record)
    '''
    return image_dataset

def load_scannet_json(json_file, image_root, dataset_name=None):
    image_dataset = load_coco_json(
        json_file, image_root, dataset_name, extra_annotation_keys=['semantic_instance_id_scannet'])
    video_dataset = aggregate_images_by_sceneid(image_dataset)
    return video_dataset

def register_scannet_context_instances(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_scannet_json(json_file, image_root, name))

    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="scannet", **metadata
    )

if __name__ == "__main__":
    """
    Test Scannet json dataloader
    """
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer

    logger = setup_logger(name=__name__)
    meta = MetadataCatalog.get("scannet_train")

    json_file = "/home/pkatara/SEMSEG/scannet_two_scene.coco.json"
    image_root = "/home/pkatara/SEMSEG/scannet_frames_25k"
    dicts = load_scannet_json(json_file, image_root, dataset_name="scannet_train")
    logger.info("Done loading {} samples.".format(len(dicts)))