_base_ = [
    '../_base_/datasets/waterscenes.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/cyclic-40e.py'
]

# cylinder3d model
grid_shape = [480, 360, 32]
model = dict(
    type='Cylinder3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=[0, -3.14159265359, 0, 50, 3.14159265359, 256],
            max_num_points=-1,
            max_voxels=-1,
        ),
    ),
    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=9,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    backbone=dict(
        type='Asymm3DSpconv',
        grid_size=grid_shape,
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)),
    decode_head=dict(
        type='Cylinder3DHead',
        channels=128,
        num_classes=8,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        loss_lovasz=dict(type='LovaszLoss', loss_weight=1.0, reduction='none'),
    ),
    train_cfg=None,
    test_cfg=dict(mode='whole'),
)

train_dataloader = dict(batch_size=8, num_workers=16)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))
