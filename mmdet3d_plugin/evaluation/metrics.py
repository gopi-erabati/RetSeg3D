# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence
import torch

import mmengine
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.evaluation import seg_eval
from mmdet3d.registry import METRICS


@METRICS.register_module()
class SegMetricCustom(BaseMetric):
    """3D semantic segmentation evaluation metric.

    Args:
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 dataset_name: str = 'semantickitti',
                 **kwargs):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        self.dataset_name = dataset_name
        super(SegMetricCustom, self).__init__(
            prefix=prefix, collect_device=collect_device)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        if (self.dataset_name == 'semantickitti' or
                self.dataset_name == 'waymo'):
            for data_sample in data_samples:
                pred_3d = data_sample['pred_pts_seg']
                eval_ann_info = data_sample['eval_ann_info']
                lidar_path = data_sample['lidar_path']
                # 'data/semantickitti/sequences/08/velodyne/000000.bin'

                # code to select points by distance for results analysis
                # pts = data_batch['inputs']['points'][0]
                # dist = (pts[:, 0:3] ** 2).sum(-1).sqrt()
                # # idx = dist < 20.0
                # # idx = torch.logical_and(dist > 20.0, dist < 50.0)
                # idx = dist > 50.0
                # pred_mask = pred_3d['pts_semantic_mask']
                # pred_mask_selected = pred_mask[idx]
                # pred_3d['pts_semantic_mask'] = pred_mask_selected
                # gt_mask = eval_ann_info['pts_semantic_mask']
                # gt_mask_selected = gt_mask[idx]
                # eval_ann_info['pts_semantic_mask'] = gt_mask_selected

                cpu_pred_3d = dict()
                for k, v in pred_3d.items():
                    if hasattr(v, 'to'):
                        cpu_pred_3d[k] = v.to('cpu').numpy()
                    else:
                        cpu_pred_3d[k] = v
                self.results.append((eval_ann_info, cpu_pred_3d, lidar_path))
        elif self.dataset_name == 'nuscenes':
            for data_sample in data_samples:
                pred_3d = data_sample['pred_pts_seg']
                eval_ann_info = data_sample['eval_ann_info']
                if 'lidar_sample_token' in data_sample:
                    lidar_sample_token = data_sample['lidar_sample_token']
                    # '80a35c14dd68408d83cf0e4f814feae4'
                else:
                    lidar_sample_token = None

                # code to select points by distance for results analysis
                # pts = data_batch['inputs']['points'][0]
                # dist = (pts[:, 0:3] ** 2).sum(-1).sqrt()
                # # idx = dist < 20.0
                # # idx = torch.logical_and(dist > 20.0, dist < 40.0)
                # idx = dist > 40.0
                # pred_mask = pred_3d['pts_semantic_mask']
                # pred_mask_selected = pred_mask[idx]
                # pred_3d['pts_semantic_mask'] = pred_mask_selected
                # gt_mask = eval_ann_info['pts_semantic_mask']
                # gt_mask_selected = gt_mask[idx]
                # eval_ann_info['pts_semantic_mask'] = gt_mask_selected

                cpu_pred_3d = dict()
                for k, v in pred_3d.items():
                    if hasattr(v, 'to'):
                        cpu_pred_3d[k] = v.to('cpu').numpy()
                    else:
                        cpu_pred_3d[k] = v
                self.results.append((eval_ann_info, cpu_pred_3d,
                                     lidar_sample_token))

    def format_results(self, results):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """

        submission_prefix = self.submission_prefix
        if submission_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            submission_prefix = osp.join(tmp_dir.name, 'results')
        mmengine.mkdir_or_exist(submission_prefix)
        if self.dataset_name == 'semantickitti':
            ignore_index = self.dataset_meta['ignore_index']
            seg_label_mapping_inv = self.dataset_meta['seg_label_mapping_inv']
            # cat2label = np.zeros(len(self.dataset_meta['label2cat'])).astype(
            #     np.int64)
            # for original_label, output_idx in self.dataset_meta['label2cat'].items(
            # ):
            #     if output_idx != ignore_index:
            #         cat2label[output_idx] = original_label

            for i, (eval_ann, result, lidar_path) in enumerate(results):
                # sample_idx = eval_ann['point_cloud']['lidar_idx']
                pred_sem_mask = result['pts_semantic_mask'].astype(np.uint32)
                pred_label = seg_label_mapping_inv[pred_sem_mask].astype(np.uint32)
                # curr_file = f'{submission_prefix}/{sample_idx}.txt'
                # np.savetxt(curr_file, pred_label, fmt='%d')
                # below for semantickitti
                seq_name = lidar_path.split('/')[3]  # '11'
                frame_name = lidar_path.split('/')[-1].split('.')[0]
                mmengine.mkdir_or_exist(osp.join(submission_prefix, "submit",
                                                 "sequences", seq_name,
                                                 "predictions"))
                pred_label.tofile(
                        osp.join(
                            submission_prefix, "submit", "sequences", seq_name,
                            "predictions", f"{frame_name}.label",
                        ))
        elif self.dataset_name == 'nuscenes':
            for i, (eval_ann, result, lidar_sample_token) in enumerate(results):
                pred_sem_mask = result['pts_semantic_mask']
                mmengine.mkdir_or_exist(osp.join(
                    submission_prefix, 'submit', 'lidarseg', 'test'
                ))
                np.array(pred_sem_mask + 1).astype(np.uint8).tofile(
                    osp.join(
                        submission_prefix, 'submit', 'lidarseg', 'test',
                        f'{lidar_sample_token}_lidarseg.bin'
                    )
                )

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return None

        label2cat = self.dataset_meta['label2cat']
        ignore_index = self.dataset_meta['ignore_index']

        gt_semantic_masks = []
        pred_semantic_masks = []

        for eval_ann, sinlge_pred_results, lidar_path in results:
            gt_semantic_masks.append(eval_ann['pts_semantic_mask'])
            pred_semantic_masks.append(
                sinlge_pred_results['pts_semantic_mask'])

        ret_dict = seg_eval(
            gt_semantic_masks,
            pred_semantic_masks,
            label2cat,
            ignore_index,
            logger=logger)

        return ret_dict
