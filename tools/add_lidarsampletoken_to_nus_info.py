import mmengine
from nuscenes.nuscenes import NuScenes

# load nus info file
info_path = '/home/gopi/PhD/datasets/nuscenes_mmdet3d130/nuscenes_infos_test.pkl'
info_dict = mmengine.load(info_path)
infos = info_dict['data_list']

root_path = '/home/gopi/PhD/datasets/nuscenes_mmdet3d130'
nusc = NuScenes(version='v1.0-test', dataroot=root_path, verbose=True)

# for each sample save token and data.LIDAR_TOP in two lists
token_nus = []
lidar_sample_token_nus = []
for sample in mmengine.track_iter_progress(nusc.sample):
    token_s = sample['token']
    lidar_sample_token_s = sample['data']['LIDAR_TOP']
    token_nus.append(token_s)
    lidar_sample_token_nus.append(lidar_sample_token_s)

for info in mmengine.track_iter_progress(infos):
    token = info['token']

    # get the index of token
    token_idx = token_nus.index(token)

    # get lidar sample token at this index
    lidar_sample_token = lidar_sample_token_nus[token_idx]

    # put the lidar_sample_token at info[lidar_points']['lidar_sample_token']
    info['lidar_points']['lidar_sample_token'] = lidar_sample_token


info_dict['data_list'] = infos

mmengine.dump(info_dict, '/home/gopi/PhD/datasets/nuscenes_mmdet3d130'
                         '/nuscenes_infos_test_wlidartoken.pkl')
