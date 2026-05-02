from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, PointData
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from mmdet3d.structures.det3d_data_sample import SampleList
from .ops import Voxelization


@MODELS.register_module()
class BEVFusionSeg(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        extra_scatter:Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        pts_middle_decoder: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        seg_head_2d: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        # self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.extra_scatter = MODELS.build(
            extra_scatter) if extra_scatter is not None else None
        self.compress = nn.Conv2d(
            pts_backbone.get('in_channels', 128) + extra_scatter.get('out_channels', 16), 
            pts_backbone.get('in_channels', 128), 3, padding=1
        ) if extra_scatter is not None else None

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.pts_middle_decoder = MODELS.build(pts_middle_decoder)

        self.bbox_head = MODELS.build(
            bbox_head) if bbox_head is not None else None
        self.seg_head = MODELS.build(
            seg_head) if seg_head is not None else None
        
        if view_transform is not None:
            self.depth_loss = True if view_transform.get(
                'loss_depth_weight', None) else None
        else:
            self.depth_loss = None

        self.seg_head_2d = MODELS.build(
            seg_head_2d) if seg_head_2d is not None else None

        self.init_weights()

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_extra_scatter(self):
        """bool: Whether the detector has a extra point scatter."""
        return hasattr(self, 'extra_scatter') and self.extra_scatter is not None

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None
    
    @property
    def with_seg_head_2d(self):
        """bool: Whether the detector has a 2D segmentation head.
        """
        return hasattr(self, 'seg_head_2d') and self.seg_head_2d is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        depth,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        multi_view_img_features = x

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
                depth=depth,
            )
        return x, multi_view_img_features

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        # print('points num:', len(points[0]))
        voxel_dict = batch_inputs_dict['voxels']
        feats, coords = voxel_dict['voxels'], voxel_dict['coors']
        batch_size = coords[-1, 0] + 1

        # rcs scatter
        if self.with_extra_scatter:
            rcs_bev_feats = self.extra_scatter(
                points_coords=feats[:, 0:2],
                rcs=feats[:, 3:4],
                voxel_coords=coords,
                batch_size=batch_size
            )

        # print('feats num:', len(feats), 'coords num:', len(coords))
        x, encode_features, sparse_x = self.pts_middle_encoder(feats, coords, batch_size)
        
        if self.with_extra_scatter:
            x = self.compress(torch.cat([x, rcs_bev_feats], dim=1))
        
        return x, encode_features, sparse_x

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        if self.view_transform is not None:
            batch_input_depth = [
                    data_sample.gt_instances.depth.unsqueeze(0) 
                    for data_sample in batch_data_samples
                ]
            batch_input_depth = torch.cat(batch_input_depth, dim=0)
        else:
            batch_input_depth = None
        feats, _ = self.extract_feat(batch_inputs_dict, batch_input_metas,
                                  batch_input_depth)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats.features, batch_input_metas)
            
            res = self.add_pred_to_datasample(batch_data_samples, outputs)
            return res
        
        if self.with_seg_head:
            seg_logits_list = self.seg_head.predict(feats.features, batch_data_samples)
            for i in range(len(seg_logits_list)):
                seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)
            return self.postprocess_result(seg_logits_list, batch_data_samples)


    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        batch_input_depth,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature, mv_img_feats = self.extract_img_feat(
                imgs, deepcopy(points), lidar2image, camera_intrinsics,
                camera2lidar, img_aug_matrix, lidar_aug_matrix, batch_input_metas,
                batch_input_depth)
            features.append(img_feature)
            # Vis image feature
            # print(img_feature.shape)
            # img_feature_vis = torch.sum(img_feature[0], dim=0)
            # import mmcv
            # mmcv.imwrite(img_feature_vis.cpu().detach().numpy(), 'output.jpg')
        pts_feature, encode_features, sparse_pts_feature = self.extract_pts_feat(batch_inputs_dict)
        # print('point feature shape:', pts_feature.shape)
        features.append(pts_feature)

        if self.fusion_layer is not None:
            # Non residual connection
            x = self.fusion_layer(features)

            # Residual connection (useless)
            # fusion_feature = self.fusion_layer(features)
            # x = fusion_feature + pts_feature
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        # for i in x: print(i.shape)
        x = self.pts_neck(x)
        # for i in x: print(i.shape)
        x = self.pts_middle_decoder(x[0], sparse_pts_feature, encode_features)

        if imgs is not None:
            return x, mv_img_feats
        else:
            return x, None

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        # gt_semantic_segs = [
        #     data_sample.gt_pts_seg.voxel_semantic_mask
        #     for data_sample in batch_data_samples
        # ]
        # seg_label = torch.cat(gt_semantic_segs)
        # print('label num:', len(seg_label))
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        if self.view_transform is not None:
            batch_input_depth = [
                    data_sample.gt_instances.depth.unsqueeze(0) 
                    for data_sample in batch_data_samples
                ]
            batch_input_depth = torch.cat(batch_input_depth, dim=0)
        else:
            batch_input_depth = None
        feats, mv_img_feats = self.extract_feat(batch_inputs_dict, batch_input_metas, 
                                 batch_input_depth)

        losses = dict()
        if self.with_seg_head_2d:
            seg_2d_loss = self.seg_head_2d.loss(mv_img_feats, batch_data_samples)
            losses.update(seg_2d_loss)
        if self.depth_loss:
            gt_depth = batch_input_depth
            losses['loss_depth'] = self.view_transform.get_depth_loss(gt_depth)
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats.features, batch_data_samples)
            losses.update(bbox_loss)
        if self.with_seg_head:
            seg_loss = self.seg_head.loss(feats.features, batch_data_samples, 
                                          train_cfg=dict())
            losses.update(seg_loss)

        # losses.update(bbox_loss)

        return losses
    
    def postprocess_result(self, seg_logits_list: List[Tensor],
                           batch_data_samples: SampleList) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Args:
            seg_logits_list (List[Tensor]): List of segmentation results,
                seg_logits from model of each input point clouds sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            List[:obj:`Det3DDataSample`]: Segmentation results of the input
            points. Each Det3DDataSample usually contains:

            - ``pred_pts_seg`` (PointData): Prediction of 3D semantic
              segmentation.
            - ``pts_seg_logits`` (PointData): Predicted logits of 3D semantic
              segmentation before normalization.
        """

        for i in range(len(seg_logits_list)):
            seg_logits = seg_logits_list[i]
            seg_pred = seg_logits.argmax(dim=0)
            batch_data_samples[i].set_data({
                'pts_seg_logits':
                PointData(**{'pts_seg_logits': seg_logits}),
                'pred_pts_seg':
                PointData(**{'pts_semantic_mask': seg_pred})
            })
        return batch_data_samples


@MODELS.register_module()
class CylinderFusionSeg(BEVFusionSeg):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        extra_scatter: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        pts_middle_decoder: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        seg_head_2d: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, 
            pts_voxel_encoder=pts_voxel_encoder,
            extra_scatter=extra_scatter,
            pts_middle_encoder=pts_middle_encoder,
            fusion_layer=fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            pts_middle_decoder=pts_middle_decoder,
            view_transform=view_transform,
            img_neck=img_neck,
            pts_neck=pts_neck,
            bbox_head=bbox_head,
            init_cfg=init_cfg,
            seg_head=seg_head,
            seg_head_2d=seg_head_2d,
            **kwargs)
        
    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        # print('points num:', len(points[0]))
        voxel_dict = batch_inputs_dict['voxels']
        feats, coords = voxel_dict['voxels'], voxel_dict['coors']

        # rcs scatter
        if self.with_extra_scatter:
            rcs_bev_feats = self.extra_scatter(
                points_coords=feats[:, 0:2],
                rcs=feats[:, 5:6],
                voxel_coords=coords,
                batch_size=coords[-1, 0] + 1
            )

        feats, coords = self.pts_voxel_encoder(feats, coords)
        batch_inputs_dict['voxels']['voxel_coors'] = coords
        batch_size = coords[-1, 0] + 1
        # print('feats num:', len(feats), 'coords num:', len(coords))
        x, encode_features, sparse_x = self.pts_middle_encoder(feats, coords, batch_size)

        if self.with_extra_scatter:
            x = self.compress(torch.cat([x, rcs_bev_feats], dim=1))

        return x, encode_features, sparse_x
    
    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        if self.view_transform is not None:
            batch_input_depth = [
                    data_sample.gt_instances.depth.unsqueeze(0) 
                    for data_sample in batch_data_samples
                ]
            batch_input_depth = torch.cat(batch_input_depth, dim=0)
        else:
            batch_input_depth = None
        feats, _ = self.extract_feat(batch_inputs_dict, batch_input_metas,
                                  batch_input_depth)
        
        if self.with_seg_head:
            seg_logits_list = self.seg_head.predict(feats, batch_inputs_dict,
                                                    batch_data_samples)
            
            for i in range(len(seg_logits_list)):
                seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)
            return self.postprocess_result(seg_logits_list, batch_data_samples)
        
    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        # gt_semantic_segs = [
        #     data_sample.gt_pts_seg.voxel_semantic_mask
        #     for data_sample in batch_data_samples
        # ]
        # seg_label = torch.cat(gt_semantic_segs)
        # print('label num:', len(seg_label))
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        if self.view_transform is not None:
            batch_input_depth = [
                    data_sample.gt_instances.depth.unsqueeze(0) 
                    for data_sample in batch_data_samples
                ]
            batch_input_depth = torch.cat(batch_input_depth, dim=0)
        else:
            batch_input_depth = None
        feats, mv_img_feats = self.extract_feat(batch_inputs_dict, batch_input_metas, 
                                 batch_input_depth)

        losses = dict()
        if self.with_seg_head_2d:
            seg_2d_loss = self.seg_head_2d.loss(mv_img_feats, batch_data_samples)
            losses.update(seg_2d_loss)
        if self.depth_loss:
            gt_depth = batch_input_depth
            losses['loss_depth'] = self.view_transform.get_depth_loss(gt_depth)
        if self.with_seg_head:
            seg_loss = self.seg_head.loss(feats, batch_data_samples, 
                                          train_cfg=dict())
            losses.update(seg_loss)

        return losses


@MODELS.register_module()
class DFSCylinderFusionSeg(CylinderFusionSeg):
    """
    DFS --> Dynamic Feature Selection
    """

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        pts_middle_decoder: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        seg_head_2d: Optional[dict] = None,
        uf_block: Optional[dict] = None,
        dist_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, 
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            fusion_layer=fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            pts_middle_decoder=pts_middle_decoder,
            view_transform=view_transform,
            img_neck=img_neck,
            pts_neck=pts_neck,
            bbox_head=bbox_head,
            init_cfg=init_cfg,
            seg_head=seg_head,
            seg_head_2d=seg_head_2d,
            **kwargs)
        
        self.dist_head = MODELS.build(
            dist_head) if dist_head is not None else None
        self.with_dist_loss = dist_head.pop('with_dist_loss')

        self.ufblock = MODELS.build(
            uf_block) if uf_block is not None else None

    @property
    def with_dist_head(self):
        """bool: Whether the model has a dist head."""
        return hasattr(self, 'dist_head') and self.dist_head is not None
    
    @property
    def with_ufblock(self):
        """bool: Whether the model has a unified block."""
        return hasattr(self, 'uf_block') and self.uf_block is not None
    
    def extract_sparse_feat_list(self, sparse_feat):
        coors = sparse_feat.indices.int()
        feats = sparse_feat.features
        batch_indices = coors[:, 0].type(torch.long)
        batch_size = batch_indices[-1] + 1

        sparse_feat_list = []
        for i in range(batch_size):
            idx = (batch_indices == i)
            sparse_feat_list.append(feats[idx].unsqueeze(0))

        return sparse_feat_list
        
    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        batch_input_depth,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature, mv_img_feats = self.extract_img_feat(
                imgs, deepcopy(points), lidar2image, camera_intrinsics,
                camera2lidar, img_aug_matrix, lidar_aug_matrix, batch_input_metas,
                batch_input_depth)
            features.append(img_feature)
        pts_feature, encode_features, sparse_pts_feature = self.extract_pts_feat(batch_inputs_dict)
        # print('point feature shape:', pts_feature.shape)
        features.append(pts_feature)

        # calculate the uncertainty of the features
        if self.with_dist_loss:
            uncertainty = []

            img_3d_encoded_feat = self.pts_backbone(img_feature)
            img_3d_encoded_feat = self.pts_neck(img_3d_encoded_feat)
            img_3d_decoded_feat = self.pts_middle_decoder(
                img_3d_encoded_feat[0], sparse_pts_feature, encode_features)
            # img_3d_decoded_feat_list = self.extract_sparse_feat_list(img_3d_decoded_feat)
            img_uncertainty = self.dist_head(img_3d_encoded_feat[0])
            # print('image predict mIoU:', img_uncertainty)

            pts_3d_encoded_feat = self.pts_backbone(pts_feature)
            pts_3d_encoded_feat = self.pts_neck(pts_3d_encoded_feat)
            pts_3d_decoded_feat = self.pts_middle_decoder(
                pts_3d_encoded_feat[0], sparse_pts_feature, encode_features)
            # pts_3d_decoded_feat_list = self.extract_sparse_feat_list(pts_3d_decoded_feat)
            pts_uncertainty = self.dist_head(pts_3d_encoded_feat[0])
            # print('points predict mIoU:', pts_uncertainty)

            sum_uncertainty = img_uncertainty + pts_uncertainty
            uncertainty.append(img_uncertainty / sum_uncertainty * 2)
            uncertainty.append(pts_uncertainty / sum_uncertainty * 2)

        if self.with_ufblock:
            features = self.ufblock(features)

        if self.fusion_layer is not None:
            if self.with_dist_loss:
                camera_features, lidar_features = features
                camera_uncertainty, lidar_uncertainty = uncertainty
                B, C, H, W = lidar_features.shape
                camera_uncertainty = camera_uncertainty.view(B, 1, 1, 1).expand(B, C, H, W)
                lidar_uncertainty = lidar_uncertainty.view(B, 1, 1, 1).expand(B, C, H, W)
                camera_features = camera_uncertainty * camera_features
                lidar_features = lidar_uncertainty * lidar_features

            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        x = self.pts_middle_decoder(x[0], sparse_pts_feature, encode_features)

        if imgs is not None:
            if self.with_dist_loss:
                return x, mv_img_feats, ((img_3d_encoded_feat[0], img_3d_decoded_feat), (pts_3d_encoded_feat[0], pts_3d_decoded_feat))
            else:
                return x, mv_img_feats, None
        else:
            if self.with_dist_loss:
                return x, None, ((img_3d_encoded_feat[0], img_3d_decoded_feat), (pts_3d_encoded_feat[0], pts_3d_decoded_feat))
            else:
                return x, None, None
        
    
    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        if self.view_transform is not None:
            batch_input_depth = [
                    data_sample.gt_instances.depth.unsqueeze(0) 
                    for data_sample in batch_data_samples
                ]
            batch_input_depth = torch.cat(batch_input_depth, dim=0)
        else:
            batch_input_depth = None
        feats, mv_img_feats, non_fused_feats = self.extract_feat(batch_inputs_dict, 
                                                                 batch_input_metas, 
                                                                 batch_input_depth)

        losses = dict()
        if self.with_seg_head_2d:
            seg_2d_loss = self.seg_head_2d.loss(mv_img_feats, batch_data_samples)
            losses.update(seg_2d_loss)
        if self.depth_loss:
            gt_depth = batch_input_depth
            losses['loss_depth'] = self.view_transform.get_depth_loss(gt_depth)
        if self.with_seg_head:
            seg_loss = self.seg_head.loss(feats,
                                          batch_data_samples, 
                                          train_cfg=dict())
            
            # Shared sparse decoder decoding fused feature and single modality feature
            # Loss = Fused Feature Loss + 0.1 * Ridar Feature Loss + 0.1 * Image Feature Loss
            if self.with_dist_loss:
                for feats in non_fused_feats:
                    dense_feat, sparse_feat = feats
                    seg_loss_single = self.seg_head.loss(sparse_feat,
                                                        batch_data_samples,
                                                        train_cfg=dict())
                    for key in seg_loss.keys():
                        seg_loss[key] += 0.1 * seg_loss_single[key]

            losses.update(seg_loss)

        if self.with_dist_loss:
            # logits = self.seg_head.cls_seg(feats)
            # lovasz_loss_per_sample = self.dist_head.cal_gt(logits, batch_inputs_dict,
            #                                         batch_data_samples, 
            #                                         train_cfg=dict())
            # miou_per_sample = 1 - lovasz_loss_per_sample
            # # feats_list = self.extract_sparse_feat_list(feats)
            # _, attenn_weight = self.dist_head(bev_x)
            # dist_loss = self.dist_head.loss(attenn_weight, miou_per_sample.detach())

            dist_loss = 0
            for feats in non_fused_feats:
                dense_feat, sparse_feat = feats
                logits = self.seg_head.cls_seg(sparse_feat)
                lovasz_loss_per_sample = self.dist_head.cal_gt(logits, batch_inputs_dict,
                                                        batch_data_samples, 
                                                        train_cfg=dict())
                miou_per_sample = 1 - lovasz_loss_per_sample
                # feats_list = self.extract_sparse_feat_list(feat)
                dist_loss += self.dist_head.loss(dense_feat, miou_per_sample.detach())

            losses['loss_dist'] = dist_loss

        return losses
    
    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        if self.view_transform is not None:
            batch_input_depth = [
                    data_sample.gt_instances.depth.unsqueeze(0) 
                    for data_sample in batch_data_samples
                ]
            batch_input_depth = torch.cat(batch_input_depth, dim=0)
        else:
            batch_input_depth = None
        feats, _, _ = self.extract_feat(batch_inputs_dict, batch_input_metas,
                                  batch_input_depth)
        
        if self.with_seg_head:
            seg_logits_list = self.seg_head.predict(feats, batch_inputs_dict,
                                                    batch_data_samples)
            
            for i in range(len(seg_logits_list)):
                seg_logits_list[i] = seg_logits_list[i].transpose(0, 1)
            return self.postprocess_result(seg_logits_list, batch_data_samples)
        

@MODELS.register_module()
class DFSCylinderFusionSegV2(DFSCylinderFusionSeg):
    """
    DFS --> Dynamic Feature Selection
    """

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        pts_middle_decoder: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        seg_head_2d: Optional[dict] = None,
        uf_block: Optional[dict] = None,
        dist_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, 
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            fusion_layer=fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            pts_middle_decoder=pts_middle_decoder,
            view_transform=view_transform,
            img_neck=img_neck,
            pts_neck=pts_neck,
            bbox_head=bbox_head,
            init_cfg=init_cfg,
            seg_head=seg_head,
            seg_head_2d=seg_head_2d,
            uf_block=uf_block,
            dist_head=dist_head,
            **kwargs)
              
    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        batch_input_depth,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature, mv_img_feats = self.extract_img_feat(
                imgs, deepcopy(points), lidar2image, camera_intrinsics,
                camera2lidar, img_aug_matrix, lidar_aug_matrix, batch_input_metas,
                batch_input_depth)
            features.append(img_feature)
        pts_feature, encode_features, sparse_pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        if self.with_dist_loss:
            img_3d_encoded_feat = self.pts_backbone(img_feature)
            img_3d_encoded_feat = self.pts_neck(img_3d_encoded_feat)
            img_3d_decoded_feat = self.pts_middle_decoder(
                img_3d_encoded_feat[0], sparse_pts_feature, encode_features)

            pts_3d_encoded_feat = self.pts_backbone(pts_feature)
            pts_3d_encoded_feat = self.pts_neck(pts_3d_encoded_feat)
            pts_3d_decoded_feat = self.pts_middle_decoder(
                pts_3d_encoded_feat[0], sparse_pts_feature, encode_features)

        camera_features, lidar_features = features
        camera_features, camera_weight = self.dist_head(camera_features)
        lidar_features, lidar_weight = self.dist_head(lidar_features)
        features = (camera_features, lidar_features)

        if self.with_ufblock:
            features = self.ufblock(features)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        x = self.pts_middle_decoder(x[0], sparse_pts_feature, encode_features)

        if imgs is not None:
            if self.with_dist_loss:
                return x, mv_img_feats, ((camera_weight, img_3d_decoded_feat), (lidar_weight, pts_3d_decoded_feat))
            else:
                return x, mv_img_feats, None
        else:
            if self.with_dist_loss:
                return x, None, ((camera_weight, img_3d_decoded_feat), (lidar_weight, pts_3d_decoded_feat))
            else:
                return x, None, None