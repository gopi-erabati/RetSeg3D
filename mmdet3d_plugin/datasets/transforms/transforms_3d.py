from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmcv.transforms import BaseTransform, Compose
from mmengine import is_list_of
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PointClip(BaseTransform):
    """Clip points by the range.

    Required Keys:

    - points

    Modified Keys:

    - points

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range: List[float]) -> None:
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def transform(self, input_dict: dict) -> dict:
        """Transform function to clip points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' keys are
            updated in the result dict.
        """
        points = input_dict['points']  # points of LiDARPoints
        points_tensor = points.coord

        points_lower_bound = points_tensor.new_tensor(self.pcd_range[:3])
        points_upper_bound = points_tensor.new_tensor(self.pcd_range[3:])

        points_tensor = torch.clamp(points_tensor, points_lower_bound,
                                    points_upper_bound)
        points.coord = points_tensor

        input_dict['points'] = points

        return input_dict


@TRANSFORMS.register_module()
class PolarMixNuscenes(BaseTransform):
    """PolarMix data augmentation.

    The polarmix transform steps are as follows:

        1. Another random point cloud is picked by dataset.
        2. Exchange sectors of two point clouds that are cut with certain
           azimuth angles.
        3. Cut point instances from picked point cloud, rotate them by multiple
           azimuth angles, and paste the cut and rotated instances.

    Required Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)
    - dataset (:obj:`BaseDataset`)

    Modified Keys:

    - points (:obj:`BasePoints`)
    - pts_semantic_mask (np.int64)

    Args:
        instance_classes (List[int]): Semantic masks which represent the
            instance.
        swap_ratio (float): Swap ratio of two point cloud. Defaults to 0.5.
        rotate_paste_ratio (float): Rotate paste ratio. Defaults to 1.0.
        pre_transform (Sequence[dict], optional): Sequence of transform object
            or config dict to be composed. Defaults to None.
        prob (float): The transformation probability. Defaults to 1.0.
    """

    def __init__(self,
                 instance_classes: List[int],
                 swap_ratio: float = 0.5,
                 rotate_paste_ratio: float = 1.0,
                 pre_transform: Optional[Sequence[dict]] = None,
                 prob: float = 1.0) -> None:
        assert is_list_of(instance_classes, int), \
            'instance_classes should be a list of int'
        self.instance_classes = instance_classes
        self.swap_ratio = swap_ratio
        self.rotate_paste_ratio = rotate_paste_ratio

        self.prob = prob
        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    def polar_mix_transform(self, input_dict: dict, mix_results: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            mix_results (dict): Mixed dict picked from dataset.

        Returns:
            dict: output dict after transformation.
        """
        mix_points = mix_results['points']
        mix_pts_semantic_mask = mix_results['pts_semantic_mask']

        points = input_dict['points']
        pts_semantic_mask = input_dict['pts_semantic_mask']

        # 1. swap point cloud
        if np.random.random() < self.swap_ratio:
            start_angle = (np.random.random() - 1) * np.pi  # -pi~0
            end_angle = start_angle + np.pi
            # calculate horizontal angle for each point
            yaw = -torch.atan2(points.coord[:, 1], points.coord[:, 0])
            mix_yaw = -torch.atan2(mix_points.coord[:, 1], mix_points.coord[:,
                                                                            0])

            # select points in sector
            idx = (yaw <= start_angle) | (yaw >= end_angle)
            mix_idx = (mix_yaw > start_angle) & (mix_yaw < end_angle)

            # swap
            points = points.cat([points[idx], mix_points[mix_idx]])
            pts_semantic_mask = np.concatenate(
                (pts_semantic_mask[idx.numpy()],
                 mix_pts_semantic_mask[mix_idx.numpy()]),
                axis=0)

        # 2. rotate-pasting
        if np.random.random() < self.rotate_paste_ratio:
            # extract instance points
            instance_points, instance_pts_semantic_mask = [], []
            for instance_class in self.instance_classes:
                mix_idx = mix_pts_semantic_mask == instance_class
                instance_points.append(mix_points[mix_idx])
                instance_pts_semantic_mask.append(
                    mix_pts_semantic_mask[mix_idx])
            instance_points = mix_points.cat(instance_points)
            instance_pts_semantic_mask = np.concatenate(
                instance_pts_semantic_mask, axis=0)

            # rotate-copy
            copy_points = [instance_points]
            copy_pts_semantic_mask = [instance_pts_semantic_mask]
            angle_list = [
                np.random.choice([-np.pi, np.pi])
            ]
            for angle in angle_list:
                new_points = instance_points.clone()
                new_points.rotate(angle)
                copy_points.append(new_points)
                copy_pts_semantic_mask.append(instance_pts_semantic_mask)
            copy_points = instance_points.cat(copy_points)
            copy_pts_semantic_mask = np.concatenate(
                copy_pts_semantic_mask, axis=0)

            points = points.cat([points, copy_points])
            pts_semantic_mask = np.concatenate(
                (pts_semantic_mask, copy_pts_semantic_mask), axis=0)

        input_dict['points'] = points
        input_dict['pts_semantic_mask'] = pts_semantic_mask
        return input_dict

    def transform(self, input_dict: dict) -> dict:
        """PolarMix transform function.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: output dict after transformation.
        """
        if np.random.rand() > self.prob:
            return input_dict

        assert 'dataset' in input_dict, \
            '`dataset` is needed to pass through PolarMix, while not found.'
        dataset = input_dict['dataset']

        # get index of other point cloud
        index = np.random.randint(0, len(dataset))

        mix_results = dataset.get_data_info(index)

        if self.pre_transform is not None:
            # pre_transform may also require dataset
            mix_results.update({'dataset': dataset})
            # before polarmix need to go through
            # the necessary pre_transform
            mix_results = self.pre_transform(mix_results)
            mix_results.pop('dataset')

        input_dict = self.polar_mix_transform(input_dict, mix_results)

        return input_dict

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(instance_classes={self.instance_classes}, '
        repr_str += f'swap_ratio={self.swap_ratio}, '
        repr_str += f'rotate_paste_ratio={self.rotate_paste_ratio}, '
        repr_str += f'pre_transform={self.pre_transform}, '
        repr_str += f'prob={self.prob})'
        return repr_str
