import argparse
import os
from os import path as osp
from pathlib import Path
import natsort
import mmengine


def get_waymo_info(root_path, split):
    """Create info file in the form of
        data_infos={
            'metainfo': {'DATASET': 'Waymo'},
            'data_list': {
                00000: {
                    'lidar_points':{
                        'lidat_path':'training/segmentxxx/velodyne/000000.bin'
                    },
                    'pts_semantic_mask_path':
                        'training/segmentxxx/labels/000000.label'
                },
                ...
            }
        }
    """
    data_infos = dict()
    data_infos['metainfo'] = dict(DATASET='Waymo')
    data_list = []

    split_path = osp.join(root_path, split)
    split_dirs = os.listdir(split_path)
    split_dirs = natsort.natsorted(split_dirs)

    for segment_name in split_dirs:
        segment_path = osp.join(split_path, segment_name, 'velodyne')
        sample_num = len(os.listdir(segment_path))
        for i in range(sample_num):
            data_list.append({
                'lidar_points': {
                    'lidar_path':
                        osp.join(split, segment_name, 'velodyne',
                                 str(i).zfill(6) + '.bin'),
                    'num_pts_feats': 4
                },
                'pts_semantic_mask_path':
                    osp.join(split, segment_name, 'labels',
                             str(i).zfill(6) + '.label')
            })
    data_infos.update(dict(data_list=data_list))
    return data_infos


def waymo_data_prepare(root_path):
    print('Generate info.')
    save_path = Path(root_path)

    pkl_prefix = 'waymo'

    waymo_infos_train = get_waymo_info(root_path, split='training')
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'waymo info train file is saved to {filename}')
    mmengine.dump(waymo_infos_train, filename)
    waymo_infos_val = get_waymo_info(root_path, split='validation')
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'waymo info val file is saved to {filename}')
    mmengine.dump(waymo_infos_val, filename)
    waymo_infos_test = get_waymo_info(root_path, split='testing')
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'waymo info test file is saved to {filename}')
    mmengine.dump(waymo_infos_test, filename)


parser = argparse.ArgumentParser(description="Waymo info creator arg parser")
parser.add_argument('rootpath',
                    type=str,
                    help='specify the root path')
args = parser.parse_args()

if __name__ == '__main__':
    waymo_data_prepare(args.rootpath)
