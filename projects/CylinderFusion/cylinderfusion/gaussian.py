# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

def gaussian_polar_2d(
        shape: Tuple[int, int], 
        center: Tensor,
        pc_range: Tensor,
        voxel_size: Tensor,
        out_size_factor: int,
        sigma: float = 1) -> np.ndarray:
    """Generate gaussian polar map.

    Args:
        shape (Tuple[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    rho_radius, phi_radius = shape
    rho_idx, phi_idx = int(center[0]), int(center[1])
    rho = rho_idx * voxel_size[0] * out_size_factor + pc_range[0]
    phi = phi_idx * voxel_size[1] * out_size_factor + pc_range[1]

    phis_idx, rhos_idx = np.ogrid[(phi_idx - phi_radius):(phi_idx + phi_radius + 1), (rho_idx - rho_radius):(rho_idx + rho_radius + 1)]
    rhos = rhos_idx * voxel_size[0] * out_size_factor + pc_range[0]
    phis = phis_idx * voxel_size[1] * out_size_factor + pc_range[1]
    
    dist = (rhos ** 2 + rho ** 2 - 2 * rhos * rho * np.cos(phis - phi)) / ((voxel_size[0] * out_size_factor) ** 2)

    h = np.exp(-(dist) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_polar_heatmap_gaussian(heatmap: Tensor,
                          center: Tensor,
                          radius: int,
                          pc_range: Tensor,
                          voxel_size: Tensor,
                          out_size_factor: int,
                          k: int = 1) -> Tensor:
    """Get gaussian masked polar heatmap.

    Args:
        heatmap (Tensor): Heatmap to be masked.
        center (Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        k (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        Tensor: Masked heatmap.
    """
    diameter = radius * 2 + 1
    r = center[0] * voxel_size[0] * out_size_factor + pc_range[0]
    # if r == 0:
    if r <= radius:
        angle_radius = 45
    else:
        # angle_radius = int(radius * voxel_size[0] * out_size_factor / (2 * r) / \
        #                     voxel_size[1] / out_size_factor)
        # angle_radius = int(radius / (2 * r) / voxel_size[1] / out_size_factor)
        angle_radius = int(torch.arcsin(radius / r) / voxel_size[1] / out_size_factor)
        angle_radius = max(2, angle_radius)
    gaussian = gaussian_polar_2d((
        radius, angle_radius), center, pc_range, voxel_size, out_size_factor, sigma=diameter / 6)

    rho, phi = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(rho, radius), min(width - rho, radius + 1)
    top, bottom = min(phi, angle_radius), min(height - phi, angle_radius + 1)

    masked_heatmap = heatmap[phi - top:phi + bottom, rho - left:rho + right]
    masked_gaussian = torch.from_numpy(
        gaussian[angle_radius - top:angle_radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_polar_heatmap_gaussian_feat(heatmap: Tensor,
                                     center: Tensor,
                                     radius: int,
                                     feat: Tensor,
                                     k: int=1):
    """
    Get gaussian masked polar heatmap

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center polar coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked polar heatmap.
    """

    rho, phi = int(center[0]), int(center[1])

    height, width = heatmap.shape[-2:]

    left, right = min(rho, radius), min(width - rho, radius + 1)
    top, bottom = min(phi, radius), min(height - phi, radius + 1)

    heatmap[:, phi - top:phi + bottom, rho - left:rho + right] += feat.view(-1, 1, 1).expand_as(heatmap[:, phi - top:phi + bottom, rho - left:rho + right])

    return heatmap

def cylinder_gaussian_polar_2d(
        shape: Tuple[int, int], 
        center: Tensor,
        pc_range: Tensor,
        voxel_size: Tensor,
        out_size_factor: int,
        sigma: float = 1) -> np.ndarray:
    """Generate gaussian polar map.

    Args:
        shape (Tuple[int]): Shape of the map.
        sigma (float): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    rho_radius, phi_radius = shape
    rho_idx, phi_idx = int(center[0]), int(center[1])
    rho = rho_idx * voxel_size[0] * out_size_factor + pc_range[0]
    phi = phi_idx * voxel_size[1] * out_size_factor + pc_range[1]

    # phis_idx, rhos_idx = np.ogrid[(phi_idx - phi_radius):(phi_idx + phi_radius + 1), (rho_idx - rho_radius):(rho_idx + rho_radius + 1)]
    rhos_idx, phis_idx = np.ogrid[(rho_idx - rho_radius):(rho_idx + rho_radius + 1), (phi_idx - phi_radius):(phi_idx + phi_radius + 1)]
    rhos = rhos_idx * voxel_size[0] * out_size_factor + pc_range[0]
    phis = phis_idx * voxel_size[1] * out_size_factor + pc_range[1]
    
    dist = (rhos ** 2 + rho ** 2 - 2 * rhos * rho * np.cos(phis - phi)) / ((voxel_size[0] * out_size_factor) ** 2)

    h = np.exp(-(dist) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_cylinder_polar_heatmap_gaussian(heatmap: Tensor,
                                         center: Tensor,
                                         radius: int,
                                         pc_range: Tensor,
                                         voxel_size: Tensor,
                                         out_size_factor: int,
                                         k: int = 1) -> Tensor:
    """Get gaussian masked polar heatmap.

    Args:
        heatmap (Tensor): Heatmap to be masked.
        center (Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        k (int): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        Tensor: Masked heatmap.
    """
    diameter = radius * 2 + 1
    r = center[0] * voxel_size[0] * out_size_factor + pc_range[0]
    # if r == 0:
    if r <= radius:
        angle_radius = 45
    else:
        # angle_radius = int(radius * voxel_size[0] * out_size_factor / (2 * r) / \
        #                     voxel_size[1] / out_size_factor)
        # angle_radius = int(radius / (2 * r) / voxel_size[1] / out_size_factor)
        angle_radius = int(torch.arcsin(radius / r) / voxel_size[1] / out_size_factor)
        angle_radius = max(2, angle_radius)
    gaussian = cylinder_gaussian_polar_2d((
        radius, angle_radius), center, pc_range, voxel_size, out_size_factor, sigma=diameter / 6)

    rho, phi = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(phi, angle_radius), min(width - phi, angle_radius + 1)
    top, bottom = min(rho, radius), min(height - rho, radius + 1)

    masked_heatmap = heatmap[rho - top:rho + bottom, phi - left:phi + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 angle_radius - left:angle_radius + right]).to(heatmap.device,
                                                            torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap