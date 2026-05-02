import copy

import torch
from mmcv.cnn import ConvModule
from torch import nn
from typing import List

from mmdet3d.registry import MODELS


@MODELS.register_module()
class UFBlock(nn.Module):
    def __init__(
            self,
            img_in_channels: int, 
            radar_in_channels: int,
            hidden_channels: int = None):
        super(UFBlock, self).__init__()
        
        if hidden_channels == None:
            hidden_channels = min(img_in_channels, radar_in_channels)
        
        self.img_channel_unifier = nn.ModuleList()
        self.img_channel_unifier.append(
            ConvModule(
                radar_in_channels,
                hidden_channels,
                kernel_size=1,
                padding=0,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)
        )
        self.img_channel_unifier.append(
            ConvModule(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)
        )

        self.radar_channel_unifier = copy.deepcopy(self.img_channel_unifier)

        self.shared_feature_encoder = nn.ModuleList()
        self.shared_feature_encoder.append(
            ConvModule(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)
        )
        self.shared_feature_encoder.append(
            ConvModule(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                inplace=False)
        )
        self.shared_feature_encoder_relu = nn.ReLU(True)

        self.img_softmax = nn.Softmax(dim=1)
        self.radar_softmax = nn.Softmax(dim=1)

    def forward(self, feats: List[torch.Tensor]):
        img_feat, radar_feat = feats

        for block in self.img_channel_unifier:
            img_feat = block(img_feat)
        middle_img_feat = img_feat

        for block in self.radar_channel_unifier:
            radar_feat = block(radar_feat)
        middle_radar_feat = radar_feat

        for block in self.shared_feature_encoder:
            img_feat = block(img_feat)
            radar_feat = block(radar_feat)

        img_feat = middle_img_feat + img_feat
        radar_feat = middle_radar_feat + radar_feat

        img_feat = self.shared_feature_encoder_relu(img_feat)
        radar_feat = self.shared_feature_encoder_relu(radar_feat)

        img_weight = self.img_softmax(img_feat)
        radar_weight = self.radar_softmax(radar_feat)

        img_feat = img_weight * img_feat
        radar_feat = radar_weight * radar_feat

        return (img_feat, radar_feat)