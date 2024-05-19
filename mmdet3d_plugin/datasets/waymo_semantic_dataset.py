# Modified from mmdet3d/datasets/semantic_kitti.py

from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.datasets.seg3d_dataset import Seg3DDataset


@DATASETS.register_module()
class WaymoSemanticDataset(Seg3DDataset):
    r"""Waymo Semantic Dataset.

    This class serves as the API for experiments on the Waymo Dataset

    Args:
        data_root (str, optional): Path of dataset root. Defaults to None.
        ann_file (str): Path of annotation file. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='',
                 img='',
                 pts_instance_mask='',
                 pts_semantic_mask='').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input,
            it usually has following keys:

                - use_camera: bool
                - use_lidar: bool
            Defaults to dict(use_lidar=True, use_camera=False).
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.classes) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (np.ndarray or str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """
    METAINFO = {
        'classes': ("Car", "Truck", "Bus",
                    "Other Vehicle",
                    # Other small vehicles (e.g. pedicab) and large vehicles
                    # (e.g. construction vehicles, RV, limo, tram).
                    "Motorcyclist", "Bicyclist", "Pedestrian", "Sign",
                    "Traffic Light",
                    # Lamp post, traffic sign pole etc.
                    "Pole",
                    # Construction cone/pole.
                    "Construction Cone", "Bicycle", "Motorcycle",
                    "Building",
                    # Bushes, tree branches, tall grasses, flowers etc.
                    "Vegetation",
                    "Tree Trunk",
                    # Curb on the edge of roads. This does not include road
                    # boundaries if there’s no curb.
                    "Curb",
                    # Surface a vehicle could drive on. This includes the
                    # driveway connecting
                    # parking lot and road over a section of sidewalk.
                    "Road",
                    # Marking on the road that’s specifically for defining
                    # lanes such as
                    # single/double white/yellow lines.
                    "Lane Marker",
                    # Marking on the road other than lane markers, bumps,
                    # cateyes, railtracks etc.
                    "Other Ground",
                    # Most horizontal surface that’s not drivable,
                    # e.g. grassy hill, pedestrian walkway stairs etc.
                    "Walkable",
                    # Nicely paved walkable surface when pedestrians most
                    # likely to walk on.
                    "Sidewalk",),
        'palette': [[0, 0, 230], [112, 128, 144], [255, 127, 80],
                    [255, 158, 0], [233, 150, 70], [255, 61, 99],
                    [128, 0, 128], [47, 79, 79], [255, 140, 0],
                    [255, 99, 71], [0, 207, 191], [175, 0, 75], [75, 0, 75],
                    [222, 184, 135], [0, 175, 0], [150, 75, 0],
                    [100, 150, 245], [100, 230, 245], [30, 60, 150],
                    [80, 30, 180], [100, 80, 250], [112, 180, 60],],
        # Slategrey, Crimson, Coral, Orange, Darksalmon, Red, Blue,
        # Darkslategrey, Dark orange, Tomato, nuTomy green, -, - , -,
        # Burlywood, Green
        'seg_valid_class_ids':
            tuple(range(22)),
        'seg_all_class_ids':
            tuple(range(22)),
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='',
                     img='',
                     pts_instance_mask='',
                     pts_semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 ignore_index: Optional[int] = None,
                 scene_idxs: Optional[Union[str, np.ndarray]] = None,
                 test_mode: bool = False,
                 **kwargs) -> None:
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            test_mode=test_mode,
            **kwargs)

    def get_seg_label_mapping(self, metainfo):
        seg_label_mapping = np.zeros(metainfo['max_label'] + 1, dtype=np.int64)
        for idx in metainfo['seg_label_mapping']:
            seg_label_mapping[idx] = metainfo['seg_label_mapping'][idx]
        return seg_label_mapping
