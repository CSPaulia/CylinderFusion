_base_ = [
    './cylinderfusionseg_radar_voxel720-720-32_second_secfpn_6xb16-cyclic-50e_xyz2xzy_waterscenes.py'
]

input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

model = dict(
    type='CylinderFusionSeg',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[156.5110, 159.9252, 154.3424],
        std=[57.6876, 57.3631, 62.7212],
        bgr_to_rgb=False),
    extra_scatter=dict(
        type='CylinderScatterRCS',
        out_channels=16, 
        out_shapes=[720, 720], 
        point_cloud_range=[0, 0, -32, 200, 3.14159265359, 32],
        voxel_size=[0.2777777778, 0.00436332313, 2.0],
        downsample=8),
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa: E501
        )),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    view_transform=dict(
        type='CylinderDepthLSSTransform',
        in_channels=256,
        out_channels=128,
        image_size=[544, 960],
        feature_size=[68, 120],
        rho_bound=[0.0, 200.0, 1.11111111],
        phi_bound=[0.0, 3.14159265359, 0.0174533],
        zbound=[-32.0, 32.0, 64.0],
        dbound=[1.0, 200.0, 1.0],
        downsample=2),
    fusion_layer=dict(
        type='DCALearnedPEFuser', 
        img_in_channels=128,
        radar_in_channels=128,  
        out_channels=128, 
        bev_size=[90, 90],
        transformer_config=dict(
            n_levels=3,
            n_heads=8, # Can be divided by in_chanels[0]
            dim_ffn=128,
            dropout=0.3,
            n_points=15,
        )),
    # seg_head_2d=dict(
    #     type='UpperHead',
    #     in_channel=256,
    #     num_classes=9,
    #     loss_seg=dict(
    #         type='mmdet.CrossEntropyLoss',
    #         use_sigmoid=False,
    #         loss_weight=1.0))
    )

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='BEVLoadMultiViewImageAnnotationFromFiles',
        to_float32=True,
        color_type='grayscale',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=9,
        use_dim=[0, 2, 1, 3, 4, 5, 6],
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.float32',
        dataset_type='waterscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='ImageAndAnnotationAug3D',
        final_dim=[544, 960],
        resize_lim=[0.45, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True),
    dict(
        type='PointToMultiViewDepth',
        dbound=[1.0, 200.0, 1.0],
        ignore_index=7),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[0.0, 0.0],
        translation_std=[0.1, 0.0, 0.1]),
    dict(type='BEVFusionRandomVerticalFlip3D'),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'pts_semantic_mask', 'gt_depth', 'gt_semantic_mask_2d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=9,
        use_dim=[0, 2, 1, 3, 4, 5, 6],
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.float32',
        dataset_type='waterscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='ImageAug3D',
        final_dim=[544, 960],
        resize_lim=[0.5, 0.5],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    # dict(type='ImageBlur', noise_level=1.0),
    dict(
        type='PointToMultiViewDepth',
        dbound=[1.0, 200.0, 1.0],
        ignore_index=7),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'pts_semantic_mask', 'gt_depth'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        ann_file='waterscenes_infos_with_cam_annotations_yz2zy_train.pkl',
        pipeline=train_pipeline, 
        modality=input_modality))
val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        ann_file='waterscenes_infos_with_cam_annotations_yz2zy_val.pkl',
        pipeline=test_pipeline, 
        modality=input_modality))
test_dataloader = dict(
    batch_size=4,
    num_workers=8,
    dataset=dict(
        ann_file='waterscenes_infos_with_cam_annotations_yz2zy_test.pkl',
        pipeline=test_pipeline, 
        modality=input_modality))

# learning rate
lr = 0.0001
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=lr * 1e-4,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=12,
        eta_min=1,
        begin=8,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=5))
# del _base_.custom_hooks
