import torch
import numpy as np
import torchvision.transforms.functional as torch_transforms

import itertools

import ipdb
st = ipdb.set_trace

def unproject(intrinsics, poses, depths):
    """
    Inputs:
        intrinsics: 4 X 4
        poses: B X V X 4 X 4 (torch.tensor)
        depths: B X V X H X W (torch.tensor)
    
    Outputs:
        world_coords: B X V X H X W X 3 (all valid 3D points)
        valid: B X V X H X W (bool to indicate valid points)
                can be used to index into RGB images
                to get N X 3 valid RGB values
    """
    B, V, H, W = depths.shape
    fx, fy, px, py = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    y = torch.arange(0, H)
    x = torch.arange(0, W)
    y, x = torch.meshgrid(y, x)

    x = x[None, None].repeat(B, V, 1, 1).flatten(2)
    y = y[None, None].repeat(B, V, 1, 1).flatten(2)
    z = depths.flatten(2)
    x = (x - px) * z / fx
    y = (y - py) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)

    world_coords = (poses @ cam_coords.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    world_coords = world_coords[..., :3] / world_coords[..., 3][..., None]

    world_coords = world_coords.reshape(B, V, H, W, 3)
    return world_coords

def backproject_depth(depths, poses):
    _, _, H, W = depths.shape
    intrinsics = get_scannet_intrinsic([H, W])
    xyz = unproject(intrinsics, poses, depths)
    return xyz

def backprojector(multi_scale_features, depths, poses):
    """
    Inputs:
        multi_scale_features: list
            [B*V, 256, 15, 20], [B*V, 256, 30, 40], [B*V, 256, 60, 80]
        depths: tensor [B, 5, 480, 640]
        poses: tensor [B, 5, 4, 4]
        mask_features: [B, 5, 256, 120, 160]

    Outputs:
        list: []
            B, V, H, W, 3
    """
    multi_scale_xyz = []
    xyz = backproject_depth(depths, poses)
    B, V, H, W, _ = xyz.shape
    for feat in multi_scale_features:
        h, w = feat.shape[2:]
        xyz_ = torch.nn.functional.interpolate(
            xyz.reshape(B*V, H, W, 3).permute(0, 3, 1, 2), size=(h, w),
            mode="nearest").permute(0, 2, 3, 1).reshape(B, V, h, w, 3)
        multi_scale_xyz.append(xyz_.float())
    
    return multi_scale_xyz

def get_scannet_intrinsic(image_size):
    scannet_intrinsic = np.array([[577.871,   0.       , 319.5],
                                  [  0.       , 577.871, 239.5],
                                  [  0.       ,   0.       ,   1. ],
                                ])
    scannet_intrinsic[0] /= 480 / image_size[0]
    scannet_intrinsic[1] /= 640 / image_size[1]
    return scannet_intrinsic

def convert_video_instances_to_3d(
    instances_all, num_frames, h_pad, w_pad, convert_point_semantic_instance=True
    ):
    all_instances = list(itertools.chain.from_iterable([instances.instance_ids for instances in instances_all]))
    unique_instances = torch.unique(torch.tensor(all_instances))
    inst_to_count = {inst.item(): id for id, inst in enumerate(unique_instances)}
    num_instances = len(unique_instances)
    mask_shape = [num_instances, num_frames, h_pad, w_pad]
    gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool)
    gt_classes_per_video = torch.zeros(num_instances, dtype=torch.int64)

    if convert_point_semantic_instance:
        point_semantic_instance_label_per_video = torch.zeros(
            (num_frames, h_pad, w_pad)
        )
    
    for f_i, targets_per_frame in enumerate(instances_all):
        gt_cls = targets_per_frame.gt_classes
        gt_instance_ids = targets_per_frame.instance_ids
        h, w = targets_per_frame.image_size
        for idx, instance_id in enumerate(gt_instance_ids):
            inst_idx = inst_to_count[instance_id.item()]
            gt_masks_per_video[inst_idx, f_i, :h, :w] = targets_per_frame.gt_masks[idx]
            gt_classes_per_video[inst_idx] = gt_cls[idx]
            if convert_point_semantic_instance:
                new_instance_id = (gt_cls[idx]+1) * 1000 + instance_id // 1000
                point_semantic_instance_label_per_video[f_i, :h, :w][targets_per_frame.gt_masks[idx]] = new_instance_id
    
    target_dict = {
        "labels": gt_classes_per_video,
        "masks": gt_masks_per_video.float()
    }

    if convert_point_semantic_instance:
        target_dict["point_semantic_instance_label"] = point_semantic_instance_label_per_video

    return target_dict

def convert_2d_to_3d(video_data):
    # assert batch size 1
    images = torch.stack(video_data["images"])
    depths = torch.stack(video_data["depths"]).unsqueeze(0)
    poses = torch.stack(video_data["poses"]).unsqueeze(0)
    coordinates = backprojector([images], depths, poses)[0].numpy().reshape(-1, 3)
    color = images.permute(0, 2, 3, 1).numpy().reshape(-1, 3)

    h, w = video_data["images"][0].shape[1:]
    target_dict = convert_video_instances_to_3d(
        video_data['instances_all'],
        len(video_data['instances_all']),
        h, w,
        convert_point_semantic_instance=True
    )
    valids = torch.stack(video_data["valids"]).reshape(-1, 1).numpy()
    segments = torch.stack(video_data["segments"]).reshape(-1, 1).numpy()
    
    semantic_instance = target_dict["point_semantic_instance_label"].numpy().reshape(-1)
    semantic = semantic_instance // 1000
    instance =  semantic_instance % 1000
    labels = np.vstack((semantic, instance)).transpose()
    # repeating normals as colors as mask3d doesnt use normal features.
    #print("coordinates shape: ", coordinates.shape)
    #print("color shape: ", color.shape)
    #print("segments shape: ", segments.shape)
    #print("labels shape: ", labels.shape)
    points = np.hstack([coordinates, color, color, segments, labels])
    return points, valids