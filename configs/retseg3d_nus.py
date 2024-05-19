plugin = True
plugin_dir = 'mmdet3d_plugin'

point_cloud_range = [-51.2, -51.2, -4, 51.2, 51.2, 2.4]
voxel_size = (0.1, 0.1, 0.1)
grid_size = [1024, 1024, 64]

model = dict(
    type='RetSeg3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessorCustom',
        voxel=True,
        voxel_type='minkunet',
        batch_first=True,
        max_voxels=120000,
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
            window_shape=[[3, 3, 3], [6, 6, 6], [12, 12, 4], [24, 24, 4],
                          [24, 24, 2]],
            sparse_shape=[[1024, 1024, 64],
                          [512, 512, 32], [256, 256, 16],
                          [128, 128, 8], [64, 64, 4]],
            group_size=[20, 128, 184, 128, 64,],
            # window_shape=[[6, 6, 6], [12, 12, 6], [24, 24, 4],
            #               [24, 24, 4], [24, 24, 2]],
            # sparse_shape=[[2048, 2048, 128], [1024, 1024, 64],
            #               [512, 512, 32], [256, 256, 16],
            #               [128, 128, 8]],
            # group_size=[184, 216, 256, 128, 64],
            # direction=[["x"], ["y"], ["x"], ["y"], ["x"], ],
            direction=[["x", "y"], ["x", "y"], ["x", "y"], ["x", "y"], ["x", "y"]],
            drop_path_rate=0.3
        ),
        grad_ckpt_layers=[],
        grid_size=grid_size,
    ),
    decode_head=dict(
        type='RetSeg3DHead',
        channels=32,
        num_classes=16,
        loss_decode=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0,
            avg_non_ignore=True,
        ),
        ignore_index=16,
    ),
    train_cfg=dict(),
    test_cfg=dict(),
)

# DATASET
# For Nuscenes we usually do 16-class segmentation.
# For labels_map we follow the uniform format of MMDetection & MMSegmentation
# i.e. we consider the unlabeled class as the last one, which is different
# from the original implementation of some methods e.g. Cylinder3D.
dataset_type = 'NuScenesSemanticDataset'
data_root = 'data/nuscenes_mmdet3d130/'
class_names = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade",
    "vegetation"
]
labels_map = {
    0: 16,  # "noise" mapped to "unlabelled"
    1: 16,  # "animal" mapped to "unlabelled"
    2: 6,  # "human.pedestrian.adult" mapped to "pedestrian"
    3: 6,  # "human.pedestrian.child" mapped to "pedestrian"
    4: 6,  # "human.pedestrian.construction_worker" mapped to "pedestrian"
    5: 16,  # "human.pedestrian.personal_mobility" mapped to "unlabelled"
    6: 6,  # "human.pedestrian.police_officer" mapped to "pedestrian"
    7: 16,  # "human.pedestrian.stroller" mapped to "unlabelled"
    8: 16,  # "human.pedestrian.wheelchair" mapped to "unlabelled"
    9: 0,  # "movable_object.barrier" mapped to barrier
    10: 16,  # "movable_object.debris" mapped to "unlabelled"
    11: 16,  # "movable_object.pushable_pullable" mapped to "unlabelled"
    12: 7,  # "movable_object.trafficcone" mapped to traffic cone
    13: 16,  # "static_object.bicycle_rack" mapped to "unlabelled"
    14: 1,  # "vehicle.bicycle" mapped to bicycle
    15: 2,  # "vehicle.bus.bendy" mapped to bus
    16: 2,  # "vehicle.bus.rigid" mapped to bus
    17: 3,  # "vehicle.car" mapped to car
    18: 4,  # "vehicle.construction" mapped to construction vehicle
    19: 16,  # "vehicle.emergency.ambulance" mapped to "unlabelled"
    20: 16,  # "vehicle.emergency.police" mapped to "unlabelled"
    21: 5,  # "vehicle.motorcycle" mapped to motorcycle
    22: 8,  # "vehicle.trailer" mapped to trailer
    23: 9,  # "vehicle.truck" mapped to truck
    24: 10,  # "flat.driveable_surface" mapped to drivable surface
    25: 11,  # "flat.other"
    26: 12,  # "flat.sidewalk"
    27: 13,  # "flat.terrain"
    28: 14,  # "static.manmade"
    29: 16,  # "static.other" mapped to "unlabelled"
    30: 15,  # "static.vegetation"
    31: 16  # "vehicle.ego" mapped to "unlabelled"
}

metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=31)

input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(pts='samples/LIDAR_TOP', img='', sweeps='sweeps/LIDAR_TOP',
                   pts_semantic_mask='lidarseg/v1.0-trainval')

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/semantickitti/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        dataset_type='nuscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='LaserMix',
                    num_areas=[3, 4, 5, 6],
                    pitch_angles=[-30, 10],
                    pre_transform=[
                        dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=5,
                            use_dim=4),
                        dict(
                            type='LoadAnnotations3D',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.uint8',
                            dataset_type='nuscenes',
                            backend_args=backend_args),
                        dict(type='PointSegClassMapping')
                    ],
                    prob=1)
            ],
            [
                dict(
                    type='PolarMixNuscenes',
                    instance_classes=[0, 1, 2, 3, 4, 5, 6, 8, 9],
                    swap_ratio=0.5,
                    rotate_paste_ratio=1.0,
                    pre_transform=[
                        dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=5,
                            use_dim=4),
                        dict(
                            type='LoadAnnotations3D',
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.uint8',
                            dataset_type='nuscenes',
                            backend_args=backend_args),
                        dict(type='PointSegClassMapping')
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

val_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        dataset_type='nuscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(type='PointClip', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(type='PointClip', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'],
         meta_keys=['img_path', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'num_pts_feats', 'pcd_trans',
                    'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
                    'pcd_rotation_angle', 'lidar_path',
                    'transformation_3d_flow', 'trans_mat',
                    'affine_aug', 'sweep_img_metas', 'ori_cam2img',
                    'cam2global', 'crop_offset', 'img_crop_offset',
                    'resize_img_shape', 'lidar2cam', 'ori_lidar2img',
                    'num_ref_frames', 'num_views', 'ego2global',
                    'axis_align_matrix', 'lidar_sample_token'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]
tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.uint8',
        dataset_type='nuscenes',
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
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=16,
        data_prefix=data_prefix,
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
        ann_file='nuscenes_infos_val.pkl',
        pipeline=val_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        ignore_index=16,
        data_prefix=data_prefix,
        test_mode=True,
        backend_args=backend_args))

val_dataloader = test_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = dict(type='SegMetricCustom', dataset_name='nuscenes')
# test_evaluator = dict(type='SegMetricCustom',
#                       submission_prefix='./Results'
#                                         '/RetSeg3D/nuscenes/test',
#                       dataset_name='nuscenes')

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizerCustom', vis_backends=vis_backends, name='visualizer')

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
