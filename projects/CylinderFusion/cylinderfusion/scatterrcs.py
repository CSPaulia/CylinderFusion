from copy import deepcopy
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import numpy as np

from mmdet3d.registry import MODELS

from .gaussian import draw_polar_heatmap_gaussian, draw_cylinder_polar_heatmap_gaussian, draw_polar_heatmap_gaussian_feat


def gaussian_2d(shape: Tuple[int, int], sigma: float = 1) -> np.ndarray:
    """Generate gaussian map.

    Args:
        shape (Tuple[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap: Tensor,
                          center: Tensor,
                          radius: int,
                          k: int = 1) -> Tensor:
    """Get gaussian masked heatmap.

    Args:
        heatmap (Tensor): Heatmap to be masked.
        center (Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        k (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_heatmap_gaussian_feat(heatmap, center, radius, feat, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1

    # x, y = int(center[0]), int(center[1])
    y, x = int(center[0]), int(center[1])

    height, width = heatmap.shape[-2:]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    heatmap[:, y - top:y + bottom, x - left:x + right] += feat.view(-1, 1, 1).expand_as(heatmap[:, y - top:y + bottom, x - left:x + right])

    return heatmap


@MODELS.register_module()
class ScatterRCS(nn.Module):
    def __init__(self, 
                 out_channels, 
                 out_shapes, 
                 point_cloud_range,
                 downsample=8):
        super(ScatterRCS, self).__init__()
        self.rcs_att = nn.Conv2d(2, out_channels, 1)

        self.x_range = (point_cloud_range[0], point_cloud_range[3])
        self.y_range = (point_cloud_range[1], point_cloud_range[4])

        self.nx = out_shapes[0]
        self.ny = out_shapes[1]

        self.downsample_module = nn.AdaptiveAvgPool2d(
            (self.nx//downsample, self.ny//downsample))

    def xy_norm(self, coords):
        coords[:, 0] = torch.clamp(coords[:, 0], self.x_range[0], self.x_range[1])
        coords[:, 1] = torch.clamp(coords[:, 1], self.y_range[0], self.y_range[1])
        coords[:, 0:1] = (coords[:, 0:1] - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        coords[:, 1:2] = (coords[:, 1:2] - self.y_range[0]) / (self.y_range[1] - self.y_range[0])
        
        return coords

    def forward(self, points_coords, rcs, voxel_coords, batch_size=None):
        xy = deepcopy(points_coords)
        xy_norm = self.xy_norm(xy)

        heatmap = points_coords.new_zeros((batch_size, self.ny, self.nx))
        heatmap_feat = points_coords.new_zeros((batch_size, 1, self.ny, self.nx))

        r = xy_norm[:, 0:1]**2 + xy_norm[:, 1:2]**2
        true_rcs = rcs * r
        true_rcs = nn.functional.relu(true_rcs)

        radius = true_rcs + 1

        for i in range(voxel_coords.shape[0]):
            batch, _, y, x = voxel_coords[i]
            draw_heatmap_gaussian(heatmap[batch], [x, y], int(radius[i].data.item()))
            heatmap_feat[batch] = draw_heatmap_gaussian_feat(
                heatmap_feat[batch], [x, y], int(radius[i].data.item()), rcs[i])
        rcs_att = self.rcs_att(torch.cat([heatmap.unsqueeze(dim=1), heatmap_feat],dim=1))

        rcs_att = self.downsample_module(rcs_att)
        return rcs_att
    

@MODELS.register_module()
class CylinderScatterRCS(nn.Module):
    def __init__(self, 
                 out_channels, 
                 out_shapes, 
                 point_cloud_range,
                 voxel_size,
                 downsample=8):
        super(CylinderScatterRCS, self).__init__()
        self.rcs_att = nn.Conv2d(2, out_channels, 1)

        self.pc_range = point_cloud_range
        self.rho_range = (point_cloud_range[0], point_cloud_range[3])
        self.phi_range = (point_cloud_range[1], point_cloud_range[4])

        self.voxel_size = voxel_size

        self.nx = out_shapes[0]
        self.ny = out_shapes[1]

        self.downsample_module = nn.AdaptiveAvgPool2d(
            (self.nx//downsample, self.ny//downsample))

    def rhophi_norm(self, coords):
        coords[:, 0] = torch.clamp(coords[:, 0], self.rho_range[0], self.rho_range[1])
        coords[:, 1] = torch.clamp(coords[:, 1], self.phi_range[0], self.phi_range[1])
        coords[:, 0:1] = (coords[:, 0:1] - self.rho_range[0]) / (self.rho_range[1] - self.rho_range[0])
        coords[:, 1:2] = (coords[:, 1:2] - self.phi_range[0]) / (self.phi_range[1] - self.phi_range[0])
        
        return coords

    def forward(self, points_coords, rcs, voxel_coords, batch_size=None):
        rhophi = deepcopy(points_coords)
        rhophi_norm = self.rhophi_norm(rhophi)

        heatmap = points_coords.new_zeros((batch_size, self.ny, self.nx))
        heatmap_feat = points_coords.new_zeros((batch_size, 1, self.ny, self.nx))

        r = rhophi_norm[:, 0:1]**2
        true_rcs = rcs * r
        true_rcs = nn.functional.relu(true_rcs)

        radius = true_rcs + 1

        for i in range(voxel_coords.shape[0]):
            batch, rho, phi, z = voxel_coords[i]
            draw_cylinder_polar_heatmap_gaussian(
                heatmap=heatmap[batch], 
                center=[rho, phi], 
                radius=int(radius[i].data.item()),
                pc_range=self.pc_range,
                voxel_size=self.voxel_size,
                out_size_factor=1)
            heatmap_feat[batch] = draw_heatmap_gaussian_feat(
                heatmap_feat[batch], 
                [rho, phi], 
                int(radius[i].data.item()), 
                rcs[i])
        
        rcs_att = self.rcs_att(torch.cat([heatmap.unsqueeze(dim=1), heatmap_feat],dim=1))
        rcs_att = self.downsample_module(rcs_att)
        return rcs_att