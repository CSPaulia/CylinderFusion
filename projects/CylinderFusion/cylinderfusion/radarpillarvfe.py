from typing import List

import torch
from mmcv.cnn import build_norm_layer
from torch import nn

from mmdet3d.registry import MODELS

@MODELS.register_module()
class Radar7PillarVFE(nn.Module):
    def __init__(self, 
                 voxel_size, 
                 point_cloud_range,
                 use_norm: bool = True,
                 use_xyz: bool = True,
                 use_distance: bool = True,
                 use_rcs: bool = True,
                 use_vr: bool = True,
                 use_vr_comp: bool = True,
                 use_time: bool = True,
                 use_elevation: bool = True,
                 num_filters: List[int] = [64]):
        super().__init__()

        num_point_features = 0
        self.use_norm = use_norm  # whether to use batchnorm in the PFNLayer
        self.use_xyz = use_xyz
        self.with_distance = use_distance
        self.selected_indexes = []

        self.use_RCS = use_rcs
        self.use_vr = use_vr
        self.use_vr_comp = use_vr_comp
        self.use_time = use_time
        self.use_elevation = use_elevation

        self.available_features = ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']

        num_point_features += 3  # center_x, center_y, center_z, mean_x, mean_y, mean_z, time, we need 6 new

        self.x_ind = self.available_features.index('x')
        self.y_ind = self.available_features.index('y')
        self.z_ind = self.available_features.index('z')
        self.rcs_ind = self.available_features.index('rcs')
        self.vr_ind = self.available_features.index('v_r')
        self.vr_comp_ind = self.available_features.index('v_r_comp')
        self.time_ind = self.available_features.index('time')

        if self.use_xyz:  # if x y z coordinates are used, add 3 channels and save the indexes
            num_point_features += 3  # x, y, z
            self.selected_indexes.extend((self.x_ind, self.y_ind, self.z_ind))  # adding x y z channels to the indexes

        if self.use_RCS:  # add 1 if RCS is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.rcs_ind)  # adding  RCS channels to the indexes

        if self.use_vr:  # add 1 if vr is used and save the indexes. Note, we use compensated vr!
            num_point_features += 1
            self.selected_indexes.append(self.vr_ind)  # adding  v_r_comp channels to the indexes

        if self.use_vr_comp:  # add 1 if vr is used (as proxy for sensor cue) and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.vr_comp_ind)

        if self.use_time:  # add 1 if time is used and save the indexes
            num_point_features += 1
            self.selected_indexes.append(self.time_ind)  # adding  time channel to the indexes
        
        if self.with_distance:
            num_point_features += 1

        ### LOGGING USED FEATURES ###
        print("number of point features used: " + str(num_point_features))
        print("6 of these are 2 * (x y z)  coordinates realtive to mean and center of pillars")
        print(str(len(self.selected_indexes)) + " are selected original features: ")

        for k in self.selected_indexes:
            print(str(k) + ": " + self.available_features[k])

        self.selected_indexes = torch.LongTensor(self.selected_indexes)  # turning used indexes into Tensor

        self.num_filters = num_filters
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        vfe_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            norm_layer = build_norm_layer(
                cfg=dict(type='BN1d', eps=1e-5, momentum=0.1), 
                num_features=out_filters)[1]
            if i == len(num_filters) - 2:
                vfe_layers.append(nn.Linear(in_filters, out_filters))
            else:
                vfe_layers.append(
                    nn.Sequential(
                        nn.Linear(in_filters, out_filters), norm_layer,
                        nn.ReLU(inplace=True)))
        self.vfe_layers = nn.ModuleList(vfe_layers)

        ## saving size of the voxel
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]

        ## saving offsets, start of point cloud in x, y, z + half a voxel, e.g. in y it starts around -39 m
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]  # number of outputs in last output channel

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, voxel_features, coords, **kwargs):
        ## coordinate system notes
        # x is pointing forward, y is left right, z is up down
        # spconv returns voxel_coords as  [batch_idx, z_idx, y_idx, x_idx], that is why coords is indexed backwards

        if not self.use_elevation:  # if we ignore elevation (z) and v_z
            voxel_features[:, self.z_ind] = 0  # set z to zero before doing anything

        orig_xyz = voxel_features[:, :self.z_ind + 1]  # selecting x y z

        # calculate mean of points in pillars for x y z and save the offset from the mean
        # Note: they do not take the mean directly, as each pillar is filled up with 0-s. Instead, they sum and divide by num of points
        # points_mean = orig_xyz.sum(dim=0, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1)
        # f_cluster = orig_xyz - points_mean  # offset from cluster mean

        # calculate center for each pillar and save points' offset from the center. voxel_coordinate * voxel size + offset should be the center of pillar (coords are indexed backwards)
        f_center = torch.zeros_like(orig_xyz)
        f_center[:, 0] = voxel_features[:, self.x_ind] - (
                    coords[:, 1].to(voxel_features.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = voxel_features[:, self.y_ind] - (
                    coords[:, 2].to(voxel_features.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = voxel_features[:, self.z_ind] - (
                    coords[:, 3].to(voxel_features.dtype) * self.voxel_z + self.z_offset)

        voxel_features = voxel_features[:, self.selected_indexes]  # filtering for used features

        features = [voxel_features, f_center]

        if self.with_distance:  # if with_distance is true, include range to the points as well
            points_dist = torch.norm(orig_xyz, 2, 1, keepdim=True)  # first 2: L2 norm second 2: along 2. dim
            features.append(points_dist)

        ## finishing up the feature extraction with correct shape and masking
        features = torch.cat(features, dim=-1)

        # voxel_count = features.shape[0]
        # mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        # mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        # features *= mask

        for vfe in self.vfe_layers:
            features = vfe(features)
        # features = features.squeeze()
        return features
