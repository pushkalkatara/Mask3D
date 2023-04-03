import torch
import numpy as np
import torchvision.transforms.functional as torch_transforms

import ipdb
st = ipdb.set_trace

def convert_2d_to_3d(video_data):
    st()
    points = None
    return points

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

    y = torch.arange(0, H).cuda()
    x = torch.arange(0, W).cuda()
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
