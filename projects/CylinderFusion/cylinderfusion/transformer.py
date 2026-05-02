# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models import DetrTransformerDecoderLayer
from torch import Tensor, nn

from mmdet3d.registry import MODELS

import torch
from einops import rearrange
from mmcv.cnn.bricks.activation import GELU
from mmengine.model import BaseModule
from torch import einsum

from .multi_scale_deform_attn import MSDeformAttn


class PositionEncodingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


@MODELS.register_module()
class TransformerDecoderLayer(DetrTransformerDecoderLayer):

    def __init__(self,
                 pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128),
                 **kwargs):
        super().__init__(**kwargs)
        self.self_posembed = PositionEncodingLearned(**pos_encoding_cfg)
        self.cross_posembed = PositionEncodingLearned(**pos_encoding_cfg)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        if self.self_posembed is not None and query_pos is not None:
            query_pos = self.self_posembed(query_pos).transpose(1, 2)
        else:
            query_pos = None
        if self.cross_posembed is not None and key_pos is not None:
            key_pos = self.cross_posembed(key_pos).transpose(1, 2)
        else:
            key_pos = None
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        # Note that the `value` (equal to `query`) is encoded with `query_pos`.
        # This is different from the standard DETR Decoder Layer.
        query = self.self_attn(
            query=query,
            key=query,
            value=query + query_pos,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        # Note that the `value` (equal to `key`) is encoded with `key_pos`.
        # This is different from the standard DETR Decoder Layer.
        query = self.cross_attn(
            query=query,
            key=key,
            value=key + key_pos,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        query = query.transpose(1, 2)
        return query


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y=None, **kwargs):
        if y is not None:
            return self.fn(self.norm(x), self.norm(y), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class FFN(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


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


class DeformableCrossAttention(nn.Module):

    def __init__(
        self,
        dim_model=256,
        dim_single_head=64,
        dropout=0.3,
        n_levels=3,
        n_heads=6,
        n_points=9,
        out_sample_loc=False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(
            dim_model,
            dim_single_head,
            n_levels,
            n_heads,
            n_points,
            out_sample_loc=out_sample_loc)
        self.dropout = nn.Dropout(dropout)
        self.out_sample_loc = out_sample_loc

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        src,
        query_pos=None,
        key_pos=None,
        reference_points=None,
        src_spatial_shapes=None,
        level_start_index=None,
        src_padding_mask=None,
    ):
        # cross attention
        tgt2, sampling_locations = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            self.with_pos_embed(src, key_pos),
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = self.dropout(tgt2)

        if self.out_sample_loc:
            return tgt, sampling_locations
        else:
            return tgt


class DeformableTransformerDecoder(nn.Module):
    """Deformable transformer decoder.

    Note that the ``DeformableDetrTransformerDecoder`` in MMDet has different
    interfaces in multi-head-attention which is customized here. For example,
    'embed_dims' is not a position argument in our customized multi-head-self-
    attention, but is required in MMDet. Thus, we can not directly use the
    ``DeformableDetrTransformerDecoder`` in MMDET.
    """

    def __init__(
        self,
        dim,
        n_levels=3,
        depth=2,
        n_heads=4,
        dim_single_head=32,
        dim_ffn=256,
        dropout=0.0,
        out_attention=False,
        n_points=9,
    ):
        super().__init__()
        self.out_attention = out_attention
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.n_levels = n_levels
        self.n_points = n_points

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        SelfAttention(
                            dim,
                            n_heads=n_heads,
                            dim_single_head=dim_single_head,
                            dropout=dropout,
                            out_attention=self.out_attention,
                        ),
                    ),
                    PreNorm(
                        dim,
                        DeformableCrossAttention(
                            dim,
                            dim_single_head,
                            n_levels=n_levels,
                            n_heads=n_heads,
                            dropout=dropout,
                            n_points=n_points,
                            out_sample_loc=self.out_attention,
                        ),
                    ),
                    PreNorm(dim, FFN(dim, dim_ffn, dropout=dropout)),
                ]))

    def forward(self, x, pos_embedding, src, src_spatial_shapes,
                level_start_index, center_pos):
        if self.out_attention:
            out_cross_attention_list = []
        if pos_embedding is not None:
            center_pos_embedding = pos_embedding(center_pos)
        reference_points = center_pos[:, :,
                                      None, :].repeat(1, 1, self.n_levels, 1)
        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            if self.out_attention:
                if center_pos_embedding is not None:
                    x_att, self_att = self_attn(x + center_pos_embedding)
                    x = x_att + x
                    x_att, cross_att = cross_attn(
                        x,
                        src,
                        query_pos=center_pos_embedding,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                else:
                    x_att, self_att = self_attn(x)
                    x = x_att + x
                    x_att, cross_att = cross_attn(
                        x,
                        src,
                        query_pos=None,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                out_cross_attention_list.append(cross_att)
            else:
                if center_pos_embedding is not None:
                    x_att = self_attn(x + center_pos_embedding)
                    x = x_att + x
                    x_att = cross_attn(
                        x,
                        src,
                        query_pos=center_pos_embedding,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )
                else:
                    x_att = self_attn(x)
                    x = x_att + x
                    x_att = cross_attn(
                        x,
                        src,
                        query_pos=None,
                        reference_points=reference_points,
                        src_spatial_shapes=src_spatial_shapes,
                        level_start_index=level_start_index,
                    )

            x = x_att + x
            x = ff(x) + x

        out_dict = {'fused_features': x}
        if self.out_attention:
            out_dict.update({
                'out_attention':
                torch.stack(out_cross_attention_list, dim=2)
            })
        return out_dict
    

class LearnedPositionalEncoding3D(BaseModule):
    """Position embedding with learnable embedding weights.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=128,
                 col_num_embed=128,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPositionalEncoding3D, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        pos = rearrange(pos, 'b c h w -> b (h w) c')
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str


class DeformableTransformer(nn.Module):
    """Deformable transformer.

    Note that the ``DeformableDetrTransformerDecoder`` in MMDet has different
    interfaces in multi-head-attention which is customized here. For example,
    'embed_dims' is not a position argument in our customized multi-head-self-
    attention, but is required in MMDet. Thus, we can not directly use the
    ``DeformableDetrTransformerDecoder`` in MMDET.
    """

    def __init__(
        self,
        dim,
        n_levels=3,
        n_heads=4,
        dim_single_head=32,
        dim_ffn=256,
        dropout=0.0,
        n_points=9,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.n_levels = n_levels
        self.n_points = n_points

        self.layers.append(
            nn.ModuleList([
                PreNorm(
                    dim,
                    DeformableCrossAttention(
                        dim,
                        dim_single_head,
                        n_levels=n_levels,
                        n_heads=n_heads,
                        dropout=dropout,
                        n_points=n_points,
                    ),
                ),
                PreNorm(dim, FFN(dim, dim_ffn, dropout=dropout)),
            ]))

    def forward(self, query, query_pos, reference_points, key, key_pos, key_spatial_shapes,
                level_start_index):
        reference_points = reference_points.repeat(1, 1, self.n_levels, 1)
        for i, (cross_attn, ff) in enumerate(self.layers):
            x_att = cross_attn(
                query,
                key,
                query_pos=query_pos,
                key_pos=key_pos,
                reference_points=reference_points,
                src_spatial_shapes=key_spatial_shapes,
                level_start_index=level_start_index,
            )

            x = x_att + query
            x = ff(x) + x

        return x