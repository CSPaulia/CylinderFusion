from typing import Tuple, Union, Dict, List

import torch
from torch import nn, Tensor
from einops import rearrange
from torch import einsum

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseModule, SubMConv3d
else:
    from mmcv.ops import SparseConvTensor, SparseModule, SubMConv3d

class SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 n_heads=8,
                 dim_single_head=64,
                 dropout=0.0,
                 out_attention=False):
        super().__init__()
        inner_dim = dim_single_head * n_heads
        project_out = not (n_heads == 1 and dim_single_head == dim)

        self.n_heads = n_heads
        self.scale = dim_single_head**-0.5
        self.out_attention = out_attention

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out else nn.Identity())

    def forward(self, x):
        _, _, _, h = *x.shape, self.n_heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if self.out_attention:
            return self.to_out(out), attn
        else:
            return self.to_out(out)


@MODELS.register_module()
class DistHead(nn.Module):
    # TODO: Description
    """
    """

    def __init__(self, 
                 in_channel: int,
                 hidden_channel: int = 128, 
                 loss_reg: ConfigType = dict(
                     type='mmdet.L1Loss', 
                     reduction='mean', 
                     loss_weight=0.25),
                 loss_lovasz: ConfigType = dict(
                     type='LovaszLoss', loss_weight=1.0)) -> None:
        super(DistHead, self).__init__()

        self.self_attention = SelfAttention(
            dim=in_channel,
            n_heads=2,
            dim_single_head=in_channel)

        self.fc1 = nn.Linear(in_channel, hidden_channel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channel, 1)
        self.sigmoid = nn.Sigmoid()

        self.loss_lovasz = MODELS.build(loss_lovasz)
        self.dist_loss = MODELS.build(loss_reg)

    def cal_gt(self, inputs: dict, 
               batch_inputs_dict: dict,
               batch_data_samples: SampleList,
               train_cfg: ConfigType) -> Dict[str, Tensor]:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.voxel_semantic_mask
            for data_sample in batch_data_samples
        ]
        seg_label = torch.cat(gt_semantic_segs)
        seg_logit_feat = inputs.features
        
        loss_lovasz_per_sample = []
        coors = batch_inputs_dict['voxels']['voxel_coors']
        for batch_idx in range(len(batch_data_samples)):
            seg_logit_feat_sample = seg_logit_feat[coors[:, 0] == batch_idx]
            seg_label_sample = seg_label[coors[:, 0] == batch_idx]
            if len(seg_logit_feat_sample) <= 1:
                print(seg_logit_feat_sample, seg_label_sample)
                if torch.argmax(seg_logit_feat_sample) == seg_label_sample[0]:
                    loss_lovasz_per_sample.append(0)
                else:
                    loss_lovasz_per_sample.append(1)
            else:
                loss_lovasz_per_sample.append(self.loss_lovasz(
                    seg_logit_feat_sample, seg_label_sample))
        ground_truth = torch.tensor(loss_lovasz_per_sample).view(-1, 1).to(coors.device)

        return ground_truth
    
    def forward(self, batch_inputs: List[Tensor]):
        batch_results = []
        for x in batch_inputs:
            x = self.self_attention(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            batch_results.append(torch.mean(x, dim=1))
        return torch.cat(batch_results, dim=0).to(batch_inputs[0].device)

    def loss(self, x: Union[Tensor, Tuple[Tensor]],
             metric_results: Union[Tensor, Tuple[Tensor]]):
        # TODO: Description
        """
        """
        metric_predict = self.forward(x)
        loss_dist = self.dist_loss(metric_predict, metric_results)
        
        return loss_dist
    

@MODELS.register_module()
class BEVConvmIoUHead(nn.Module):
    # TODO: Description
    """
    """

    def __init__(self, 
                 in_channel: int,
                 hidden_channel: int = 128, 
                 bev_size: List = [90, 90],
                 loss_reg: ConfigType = dict(
                     type='mmdet.L1Loss', 
                     reduction='mean', 
                     loss_weight=0.25),
                 loss_lovasz: ConfigType = dict(
                     type='LovaszLoss', loss_weight=1.0)) -> None:
        super(BEVConvmIoUHead, self).__init__()

        self.hidden_channel = hidden_channel

        conv_cfg = dict(type='Conv2d', bias=False)
        norm_cfg = dict(type='BN', eps=1e-3, momentum=0.01)

        blocks = []
        for i in range(4):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_channel,
                    hidden_channel,
                    3,
                    stride=2,
                    padding=1),
                build_norm_layer(norm_cfg, hidden_channel)[1],
                nn.ReLU(inplace=True),
            ]

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        self.bev_size = bev_size
        downsampled_bev_size = 6

        self.fc1 = nn.Linear(downsampled_bev_size * downsampled_bev_size * hidden_channel, hidden_channel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channel, 1)
        self.sigmoid = nn.Sigmoid()

        self.loss_lovasz = MODELS.build(loss_lovasz)
        self.dist_loss = MODELS.build(loss_reg)

    def cal_gt(self, inputs: dict, 
               batch_inputs_dict: dict,
               batch_data_samples: SampleList,
               train_cfg: ConfigType) -> Dict[str, Tensor]:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.voxel_semantic_mask
            for data_sample in batch_data_samples
        ]
        seg_label = torch.cat(gt_semantic_segs)
        seg_logit_feat = inputs.features
        
        loss_lovasz_per_sample = []
        coors = batch_inputs_dict['voxels']['voxel_coors']
        for batch_idx in range(len(batch_data_samples)):
            seg_logit_feat_sample = seg_logit_feat[coors[:, 0] == batch_idx]
            seg_label_sample = seg_label[coors[:, 0] == batch_idx]
            if len(seg_logit_feat_sample) <= 1:
                if torch.argmax(seg_logit_feat_sample) == seg_label_sample[0]:
                    loss_lovasz_per_sample.append(0)
                else:
                    loss_lovasz_per_sample.append(1)
            else:
                loss_lovasz_per_sample.append(self.loss_lovasz(
                    seg_logit_feat_sample, seg_label_sample))
        ground_truth = torch.tensor(loss_lovasz_per_sample).view(-1, 1).to(coors.device)

        return ground_truth
    
    def forward(self, batch_inputs: List[Tensor]):
        B, C, H, W = batch_inputs.shape
        x = batch_inputs

        for block in self.blocks:
            x = block(x)
        
        x = x.view(B, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

    def loss(self, x: Union[Tensor, Tuple[Tensor]],
             metric_results: Union[Tensor, Tuple[Tensor]]):
        # TODO: Description
        """
        """
        metric_predict = self.forward(x)
        # print('pred:', metric_predict)
        # print('gt:', metric_results)
        loss_dist = self.dist_loss(metric_predict, metric_results)
        
        return loss_dist
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
    
@MODELS.register_module()
class AttenmIoUHead(nn.Module):
    # TODO: Description
    """
    """

    def __init__(self, 
                 channel: int = 128, 
                 bev_size: List = [90, 90],
                 loss_reg: ConfigType = dict(
                     type='mmdet.L1Loss', 
                     reduction='mean', 
                     loss_weight=0.25),
                 loss_lovasz: ConfigType = dict(
                     type='LovaszLoss', loss_weight=1.0),
                 with_dist_loss: bool = True) -> None:
        super(AttenmIoUHead, self).__init__()

        self.sa = SpatialAttention(kernel_size=3)

        if with_dist_loss:
            self.fc1 = nn.Linear(bev_size[0] * bev_size[1], channel)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(channel, 1)
            self.sigmoid = nn.Sigmoid()

            self.loss_lovasz = MODELS.build(loss_lovasz)
            self.dist_loss = MODELS.build(loss_reg)

    def cal_gt(self, inputs: dict, 
               batch_inputs_dict: dict,
               batch_data_samples: SampleList,
               train_cfg: ConfigType) -> Dict[str, Tensor]:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.voxel_semantic_mask
            for data_sample in batch_data_samples
        ]
        seg_label = torch.cat(gt_semantic_segs)
        seg_logit_feat = inputs.features
        
        loss_lovasz_per_sample = []
        coors = batch_inputs_dict['voxels']['voxel_coors']
        for batch_idx in range(len(batch_data_samples)):
            seg_logit_feat_sample = seg_logit_feat[coors[:, 0] == batch_idx]
            seg_label_sample = seg_label[coors[:, 0] == batch_idx]
            if len(seg_logit_feat_sample) <= 1:
                print(seg_logit_feat_sample, seg_label_sample)
                if torch.argmax(seg_logit_feat_sample) == seg_label_sample[0]:
                    print(0)
                    loss_lovasz_per_sample.append(0)
                else:
                    print(1)
                    loss_lovasz_per_sample.append(1)
            else:
                loss_lovasz_per_sample.append(self.loss_lovasz(
                    seg_logit_feat_sample, seg_label_sample))
        ground_truth = torch.tensor(loss_lovasz_per_sample).view(-1, 1).to(coors.device)

        return ground_truth
    
    def forward(self, batch_inputs: List[Tensor]):
        x = batch_inputs

        atten_weight = self.sa(x)
        x = atten_weight * x
        
        return x, atten_weight
    
    def post(self, atten_weight: List[Tensor]):
        x = atten_weight
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def loss(self, atten_weight: Union[Tensor, Tuple[Tensor]],
             metric_results: Union[Tensor, Tuple[Tensor]]):
        # TODO: Description
        """
        """
        metric_predict = self.post(atten_weight)
        loss_dist = self.dist_loss(metric_predict, metric_results)
        
        return loss_dist