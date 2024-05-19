from __future__ import print_function
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('split', choices=['training', 'validation', 'testing'])
parser.add_argument('--out-dir', default='./tmp')
parser.add_argument('--resize', default=0.5625, type=float)
args = parser.parse_args()

url_template = ('gs://waymo_open_dataset_v_1_4_1/archived_files/{split}/{'
                'split}_%04d.tar').format(
    split=args.split)
if args.split == 'training':
    num_segs = 32
elif args.split == 'validation':
    num_segs = 8
elif args.split == 'testing':
    num_segs = 8

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

for seg_id in range(0, num_segs):
    flag = os.system('gsutil cp ' + url_template % seg_id + ' ' +
    args.out_dir)
    assert flag == 0, ('Failed to download segment %d. Make sure gsutil is '
                       'installed') % seg_id
    os.system('cd %s; tar xf %s_%04d.tar' % (args.out_dir, args.split, seg_id))
    # tfrecords = sorted(glob.glob('%s/*.tfrecord'%args.out_dir))
    os.system('cd %s; rm %s_%04d.tar' % (args.out_dir, args.split, seg_id))
    # for record in tfrecords:
    #     extract_frame(record, 'labels/%05d.txt'%clip_id,
    #     'images/%05d'%clip_id, WAYMO_CLASSES, resize_ratio=args.resize)
    #     print("Clip %d done"%clip_id)
    #     clip_id += 1
    #     os.remove(record)

    print("Segment %d done" % seg_id)
