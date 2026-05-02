from typing import Dict, List, Tuple, Union
from copy import deepcopy

import torch
from torch import nn, Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import ConfigType, OptMultiConfig
from mmdet.models.seg_heads.base_semantic_head import BaseSemanticHead
from mmdet3d.models.decode_heads.cylinder3d_head import Cylinder3DHead
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.models.decode_heads import Base3DDecodeHead

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor


@MODELS.register_module()
class UpperHead(BaseSemanticHead):
    # TODO: Description
    """
    """

    def __init__(self, 
                 in_channel: int,
                 num_classes: int, 
                 seg_rescale_factor: float = 1 / 4, 
                 loss_seg: ConfigType = dict(
                     type='CrossEntropyLoss',
                     ignore_index=255,
                     loss_weight=1.0),
                 init_cfg: OptMultiConfig = None) -> None:
        super(UpperHead, self).__init__(
            num_classes, 
            seg_rescale_factor, 
            loss_seg, 
            init_cfg)
        
        self.img_seg_up = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)
        self.img_seg_predicter = nn.Sequential(                
            nn.Conv2d(in_channel, 
                      num_classes, 
                      kernel_size=1)
        )

        self.img_seg_loss = MODELS.build(loss_seg)

    def _stack_batch_gt(self, batch_data_samples: List[Det3DDataSample]) -> Tensor:
        """Concat 2D seg mask Groud Truth."""
        batch_segmantic_mask_2d = [
            data_sample.gt_instances.semantic_mask_2d.unsqueeze(0)
            for data_sample in batch_data_samples
        ]
        return torch.cat(batch_segmantic_mask_2d, dim=0)
    
    def forward(self, x: Union[Tensor, Tuple[Tensor]]):
        B, N, C, H, W = x.shape
        img_seg_feat = x.view(B*N, C, H, W)
        img_seg_feat = self.img_seg_up(img_seg_feat)
        img_seg_predict = self.img_seg_predicter(img_seg_feat)
        
        _, C, H, W = img_seg_predict.shape
        img_seg_predict = img_seg_predict.permute(0, 2, 3, 1).contiguous().view(-1, C)
        return img_seg_predict

    def loss(self, x: Union[Tensor, Tuple[Tensor]],
             batch_data_samples: List[Det3DDataSample]):
        # TODO: Description
        """
        """

        gt_img_semantics = self._stack_batch_gt(batch_data_samples)
        gt_img_semantics = gt_img_semantics.view(-1).long()

        img_seg_predict = self.forward(x)
        loss_img_seg = self.img_seg_loss(img_seg_predict, gt_img_semantics)
        
        return {'loss_seg_2d': loss_img_seg}
    

@MODELS.register_module()
class DFSCylinder3DHead(Cylinder3DHead):

    def loss(self, inputs: dict, batch_inputs_dict: dict,
             batch_data_samples: SampleList,
             train_cfg: ConfigType) -> Dict[str, Tensor]:
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_inputs_dict,
                                   batch_data_samples)
        return losses

    def loss_by_feat(self, seg_logit: SparseConvTensor,
                     batch_inputs_dict: dict,
                     batch_data_samples: SampleList) -> dict:
        gt_semantic_segs = [
            data_sample.gt_pts_seg.voxel_semantic_mask
            for data_sample in batch_data_samples
        ]
        seg_label = torch.cat(gt_semantic_segs)
        seg_logit_feat = seg_logit.features
        loss = dict()
        loss['loss_ce'] = self.loss_ce(
            seg_logit_feat, seg_label, ignore_index=self.ignore_index)
        loss['loss_lovasz'] = self.loss_lovasz(
            seg_logit_feat, seg_label, ignore_index=self.ignore_index)
        
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
                    seg_logit_feat_sample, seg_label_sample, ignore_index=self.ignore_index))
        loss['loss_lovasz_per_sample'] = torch.tensor(loss_lovasz_per_sample).view(-1, 1).to(coors.device)

        return loss