plugin = True
plugin_dir = 'mmdet3d_plugin'

voxel_size = (0.1, 0.1, 0.1)
point_cloud_range = [-75.2, -75.2, -2.4, 75.2, 75.2, 4]
grid_size = [1504, 1504, 64]

model = dict(
    type='RetSeg3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessorCustom',
        voxel=True,
        voxel_type='minkunet',
        batch_first=True,
        max_voxels=160000,
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1)),
    ),  # voxels (BM, 4) coors (BM, 4)[b,x,y,z]
    backbone=dict(
        type='RetSeg3DBackbone',
        input_channels=4,
        encoder_channels=[32, 64, 128, 256, 256],
        block_reps=2,
        ret_cfg=dict(
            num_heads=[2, 2, 4, 4, 4],
            window_shape=[[6, 6, 6], [12, 12, 6], [24, 24, 4],
                          [48, 48, 4], [24, 24, 2]],
            sparse_shape=[[1504, 1504, 64],
                          [752, 752, 32], [376, 376, 16],
                          [188, 188, 8], [94, 94, 4]],
            group_size=[184, 216, 256, 128, 64],
            direction=[["x"], ["y"], ["x"], ["y"], ["x"], ],
            drop_path_rate=0.3
        ),
        grad_ckpt_layers=[],
        grid_size=grid_size,
    ),
    decode_head=dict(
        type='RetSeg3DHead',
        channels=32,
        num_classes=22,
        loss_decode=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1
                          ],
            loss_weight=1.0,
            avg_non_ignore=True,
        ),
        ignore_index=22,
    ),
    train_cfg=dict(),
    test_cfg=dict(),
)

# DATASET
# For Nuscenes we usually do 16-class segmentation.
# For labels_map we follow the uniform format of MMDetection & MMSegmentation
# i.e. we consider the unlabeled class as the last one, which is different
# from the original implementation of some methods e.g. Cylinder3D.
dataset_type = 'WaymoSemanticDataset'
data_root = 'data/waymo_pointcept/'
class_names = [
    "Car", "Truck", "Bus", "Other Vehicle", "Motorcyclist", "Bicyclist",
    "Pedestrian", "Sign", "Traffic Light", "Pole", "Construction Cone",
    "Bicycle", "Motorcycle", "Building", "Vegetation", "Tree Trunk",
    "Curb", "Road", "Lane Marker", "Other Ground", "Walkable", "Sidewalk",
]
labels_map = {
    0: 22,  # mapped to "unlabelled"
    1: 0,  # Car
    2: 1,  # Truck
    3: 2,  # Bus
    4: 3,  # Other vehicle
    5: 4,  # motorcyclist
    6: 5,  # bicyclist
    7: 6,  # pedestrian
    8: 7,  # sign
    9: 8,  # traffic light
    10: 9,  # pole
    11: 10,  # construction cone
    12: 11,  # bicycle
    13: 12,  # motorcycle
    14: 13,  # building
    15: 14,  # vegetation
    16: 15,  # tree trunk
    17: 16,  # curb
    18: 17,  # road
    19: 18,  # lane marker
    20: 19,  # other ground
    21: 20,  # walkable
    22: 21  # sidewalk
}

metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=22)

input_modality = dict(use_lidar=True, use_camera=False)
# data_prefix = dict(pts='samples/LIDAR_TOP', img='',
#                    sweeps='sweeps/LIDAR_TOP',
#                    pts_semantic_mask='lidarseg/v1.0-trainval')

backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        norm_intensity=True,  # tanh
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3DWaymoSemantic',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        dataset_type='waymo',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='LaserMix',
                    num_areas=[3, 4, 5, 6],
                    pitch_angles=[-17.6, 2.4],
                    pre_transform=[
                        dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4,
                            norm_intensity=True,  # tanh
                            backend_args=backend_args),
                        dict(
                            type='LoadAnnotations3DWaymoSemantic',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.int32',
                            dataset_type='waymo',
                            backend_args=backend_args),
                        dict(type='PointSegClassMapping'),
                    ],
                    prob=1)
            ],
            [
                dict(
                    type='PolarMixNuscenes',
                    instance_classes=[0, 1, 2, 3, 4, 5, 6, 11, 12, 13],
                    swap_ratio=0.5,
                    rotate_paste_ratio=1.0,
                    pre_transform=[
                        dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4,
                            norm_intensity=True,  # tanh
                            backend_args=backend_args),
                        dict(
                            type='LoadAnnotations3DWaymoSemantic',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.int32',
                            dataset_type='waymo',
                            backend_args=backend_args),
                        dict(type='PointSegClassMapping'),
                    ],
                    prob=1)
            ],
        ],
        prob=[0.5, 0.5]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type='PointClip', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        norm_intensity=True,  # tanh
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3DWaymoSemantic',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        dataset_type='waymo',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(type='PointClip', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        norm_intensity=True,  # tanh
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]
tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        norm_intensity=True,  # tanh
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3DWaymoSemantic',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        dataset_type='waymo',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(type='PointClip', point_cloud_range=point_cloud_range),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=1.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=0.),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=1.,
                flip_ratio_bev_vertical=1.)
        ],
            [
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[pcd_rotate_range, pcd_rotate_range],
                    scale_ratio_range=[
                        pcd_scale_factor, pcd_scale_factor
                    ],
                    translation_std=[0, 0, 0])
                for pcd_rotate_range in [-0.78539816, 0.0, 0.78539816]
                for pcd_scale_factor in [0.95, 1.0, 1.05]
            ], [dict(type='Pack3DDetInputs', keys=['points'])]])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=22,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='waymo_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=22,
        test_mode=True,
        backend_args=backend_args))

val_dataloader = test_dataloader

val_evaluator = dict(type='SegMetricCustom', dataset_name='waymo')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizerCustom', vis_backends=vis_backends,
    name='visualizer')

tta_model = dict(type='Seg3DTTAModel')

# optimizer
lr = 0.002  # max learning rate
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'retention_block': dict(lr_mult=0.1)
        })
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (4 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=32)

# default_runtime
default_scope = 'mmdet3d'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
