import os

from .scannet_context import (
    register_scannet_context_instances,
    _get_scannet_instances_meta
)

# ==== Predefined splits for SCANNET ===========
_PREDEFINED_SPLITS_CONTEXT_SCANNET_25K = {
    "scannet_context_instance_train_25k": (
        "scannet_frames_25k",
        "scannet_train.coco.json"
        #  "scannet_two_scene.coco.json"
    #    "scannet_train20.coco.json"
    ),
    "scannet_context_instance_val_25k": (
        "scannet_frames_25k",
        "scannet_val.coco.json"
        # "scannet_two_scene.coco.json"
    #    "scannet_train20.coco.json"
    )
}

_PREDEFINED_SPLITS_CONTEXT_SCANNET_100K = {
    "scannet_context_instance_train_100k": (
        "frames_square",
        # "scannet_random20.coco.json"
        #"scannet_train_valid.coco.json"
         "scannet_two_scene_valid.coco.json"
        # "scannet_two_scene.coco.json"
        # "scannet_two_scene.coco_mask3d.json"
        # "scannet_train.coco.json"
        # "scannet_train.coco.json"
    ),
    "scannet_context_instance_validation_100k": (
        "frames_square",
        #"scannet_val_valid.coco.json"
        "scannet_two_scene_valid.coco.json"
        #"scannet_one_scene.coco.json"
        # "scannet_val.coco.json"
        # "scannet_train.coco.json"
        # "scannet_random20.coco.json"
        # "scannet_val.coco.json"
        #"scannet_val.coco_mask3d.json"
        # "scannet_two_scene.coco.json"
        # "scannet_two_scene.coco_mask3d.json"
        # "scannet_train.coco_mask3d.json"
    ),
    "scannet_context_instance_train_eval_100k": (
        "frames_square",
        #"scannet_ten_scene.coco.json"
        "scannet_ten_scene_valid.coco.json"
    ),
}

def register_all_scannet_context_25K(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_CONTEXT_SCANNET_25K.items():
        register_scannet_context_instances(
            key,
            _get_scannet_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_scannet_context_100K(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_CONTEXT_SCANNET_100K.items():
        register_scannet_context_instances(
            key,
            _get_scannet_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_scannet_context_25K(_root)
    register_all_scannet_context_100K(_root)