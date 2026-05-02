# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
from torch import nn as nn

from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.models.middle_encoders import SparseEncoder
from mmdet3d.registry import MODELS

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential


@MODELS.register_module()
class BEVFusionSparseDecoder(nn.Module):
    r"""Sparse encoder for BEVFusion. The difference between this
    implementation and that of ``SparseEncoder`` is that the shape order of 3D
    conv is (H, W, D) in ``BEVFusionSparseEncoder`` rather than (D, H, W) in
    ``SparseEncoder``. This difference comes from the implementation of
    ``voxelization``.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
        return_middle_feats (bool): Whether output middle features.
            Default to False.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=128,
                 output_channels=16,
                 decoder_channels=((64, 64, 64), (64, 64, 64), (32, 32, 32), 
                                   (16, )),
                 decoder_paddings=((1, 1, (0, 1, 1)), (1, 1, 1), (1, 1, 1), 
                                   (1, )),
                 block_type='conv_module',
                 return_middle_feats=False,
                 skip_connection=False):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.decoder_channels = decoder_channels
        self.decoder_paddings = decoder_paddings
        self.stage_num = len(self.decoder_channels)
        self.fp16_enabled = False
        self.return_middle_feats = return_middle_feats
        self.skip_connection = skip_connection
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        self.conv_input = make_sparse_convmodule(
            in_channels,
            self.base_channels,
            kernel_size=(1, 1, 3),
            norm_cfg=norm_cfg,
            indice_key='spconv_down2',
            conv_type='SparseInverseConv3d')

        # if self.order[0] != 'conv':  # pre activate
        #     self.conv_input = make_sparse_convmodule(
        #         in_channels,
        #         self.base_channels,
        #         3,
        #         norm_cfg=norm_cfg,
        #         padding=1,
        #         indice_key='spconv_down2',
        #         conv_type='SparseInverseConv3d',
        #         order=('conv', ))
        # else:  # post activate
        #     self.conv_input = make_sparse_convmodule(
        #         in_channels,
        #         self.base_channels,
        #         3,
        #         norm_cfg=norm_cfg,
        #         padding=1,
        #         indice_key='spconv_down2',
        #         conv_type='SparseInverseConv3d')

        decoder_out_channels = self.make_decoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(
            decoder_out_channels,
            self.output_channels,
            kernel_size=3,
            norm_cfg=norm_cfg,
            padding=1,
            indice_key='subm1_up',
            conv_type='SubMConv3d')

    def forward(self, dense_bev_features, sparse_voxel_feature,
                encode_features=None):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list]: Return spatial features
                include:

            - spatial_features (torch.Tensor): Spatial features are out from
                the last layer.
            - encode_features (List[SparseConvTensor], optional): Middle layer
                output features. When self.return_middle_feats is True, the
                module returns middle features.
        """
        coors = sparse_voxel_feature.indices.int()
        # print(len(coors))
        batch_indices = coors[:, 0].type(torch.long)
        x_indices = coors[:, 1].type(torch.long)
        y_indices = coors[:, 2].type(torch.long)
        z_indices = coors[:, 3].type(torch.long)

        correspond_feats = dense_bev_features[batch_indices, :, x_indices, y_indices]
        # print('correspond feature shapa:', correspond_feats.shape)

        # input_sp_tensor = SparseConvTensor(correspond_feats, coors,
        #                                    self.sparse_shape, batch_size)
        input_sp_tensor = sparse_voxel_feature.replace_feature(correspond_feats)
        # print('sparse input shape:', input_sp_tensor.spatial_shape)
        # print('coords num:', input_sp_tensor.indices.shape)
        x = self.conv_input(input_sp_tensor)
        # print('sparse conv_input shape:', x.spatial_shape)
        # print('coords num:', x.indices.shape)

        decode_features = []
        for i, decoder_layer in enumerate(self.decoder_layers):
            if self.skip_connection:
                x = x.replace_feature(
                    x.features + encode_features[self.stage_num - i - 1].features)
            # print(x.features.shape)
            x = decoder_layer(x)
            # print('sparse encoder_layers shape:', x.spatial_shape)
            # print('coords num:', x.indices.shape)
            decode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(decode_features[-1])
        # print('sparse conv_out shape:', out.spatial_shape)
        # print('coords num:', x.indices.shape)
        return out
        
    def make_decoder_layers(
        self,
        make_block: nn.Module,
        norm_cfg: Dict,
        in_channels: int,
        block_type: Optional[str] = 'conv_module',
        conv_cfg: Optional[dict] = dict(type='SubMConv3d')
    ) -> int:
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.decoder_layers = SparseSequential()

        for i, blocks in enumerate(self.decoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.decoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != len(self.decoder_channels) - 1 and j == len(
                    blocks) - 1 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{len(self.decoder_channels) - i}',
                            conv_type='SparseInverseConv3d'))
                elif block_type == 'basicblock':
                    if j == 0 and i != 0:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                indice_key=f'spconv{len(self.decoder_channels) - i}',
                                conv_type='SparseInverseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}_up',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'decoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            self.decoder_layers.add_module(stage_name, stage_layers)
        return out_channels
