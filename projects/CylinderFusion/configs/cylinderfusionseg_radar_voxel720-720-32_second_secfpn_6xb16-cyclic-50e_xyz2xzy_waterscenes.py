_base_ = ['_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.CylinderFusion.cylinderfusion'], allow_failed_imports=False)

# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
class_names = [
    'pier', 'buoy', 'sailor', 'ship', 'boat', 'vessel', 'kayak', 'background'
]
labels_map = {
    -1: 7,  # "background"
    0: 0,  # "pier"
    1: 1,  # "buoy"
    2: 2,  # "sailor"
    3: 3,  # "ship"
    4: 4,  # "boat"
    5: 5,  # "vessel"
    6: 6,  # "kayak"
}

metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=7)
dataset_type = 'WaterScenesDataset'
data_root = '/data1/csj_data/dataset/WaterScenes/WaterScenes-Published-unzip-KITTI-format'
input_modality = dict(use_lidar=True, use_camera=False)

backend_args = None

model = dict(
    type='CylinderFusionSeg',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=[720, 720, 32],
            point_cloud_range=[0, 0, -32, 200, 3.14159265359, 32],
            max_num_points=-1,
            max_voxels=-1,
        ),
    ),
    extra_scatter=dict(
        type='CylinderScatterRCS',
        out_channels=16, 
        out_shapes=[720, 720], 
        point_cloud_range=[0, 0, -32, 200, 3.14159265359, 32],
        voxel_size=[0.2777777778, 0.00436332313, 2.0],
        downsample=8),
    pts_voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=9,
        with_voxel_center=True,
        grid_shape=[720, 720, 32],
        point_cloud_range=[0, 0, -32, 200, 3.14159265359, 32],
        feat_compression=16,
        return_point_feats=False),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=16,
        sparse_shape=[720, 720, 32],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock',
        return_middle_feats=True,
        return_sparse=True),
    pts_backbone=dict(
        type='SECOND',
        in_channels=128,
        out_channels=[32, 64],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[32, 64],
        out_channels=[64, 64],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_middle_decoder=dict(
        type='BEVFusionSparseDecoder',
        in_channels=128,
        sparse_shape=[90, 90, 1],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        decoder_channels=((128, 128), (64, 64, 64), (32, 32, 32), (16, 16, 16)),
        decoder_paddings=((0, 0), ((1, 1, 0), 0, 0), (1, 0, 0), (1, 0, 0)),
        block_type='basicblock',
        skip_connection=True),
    seg_head=dict(
        type='Cylinder3DHead',
        channels=16,
        num_classes=len(class_names),
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 1.0],
            # class_weight = [2.97367039, 4.42751549, 6.23505838, 
            #                 1.90675483, 3.83602267, 3.04136617,
            #                 9.03107735, 0.33274879],
            loss_weight=1.0),
        loss_lovasz=dict(type='LovaszLoss', loss_weight=1.0, reduction='none')),
    train_cfg=dict(),
    test_cfg=dict())

train_pipeline = [
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
        type='GlobalRotScaleTrans',
        rot_range=[0.0, 0.0],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0, 0.1]),
    dict(type='BEVFusionRandomVerticalFlip3D'),
    # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

test_pipeline = [
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
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

train_dataloader = dict(
    batch_size=16,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waterscenes_infos_with_cam_annotations_yz2zy_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=16,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waterscenes_infos_with_cam_annotations_yz2zy_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=16,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waterscenes_infos_with_cam_annotations_yz2zy_test.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        backend_args=backend_args))

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# learning rate
lr = 0.0001
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=20,
        eta_min=lr * 10,
        begin=0,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=30,
        eta_min=lr * 1e-4,
        begin=20,
        end=50,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=20,
        eta_min=0.85 / 0.95,
        begin=0,
        end=20,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=30,
        eta_min=1,
        begin=20,
        end=50,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
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
log_processor = dict(window_size=50)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=5))
