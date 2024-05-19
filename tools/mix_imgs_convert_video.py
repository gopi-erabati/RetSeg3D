import cv2
import numpy as np
import os
from os import path as osp
from tqdm import tqdm
import natsort
from PIL import Image

data_root_kitti = ('/home/gopi/PhD/workdirs/Results/RetSeg3D/semantickitti/val'
                   '/vis_gt_pred_error/')
data_root_nus = ('/home/gopi/PhD/workdirs/Results/RetSeg3D/nuscenes/val'
                 '/vis_gt_pred_error')
data_root_waymo = ('/home/gopi/PhD/workdirs/Results/RetSeg3D/waymo/val'
                   '/vis_gt_pred_error')
img_array = []

img_names_kitti = os.listdir(data_root_kitti)
img_names_kitti = natsort.natsorted(img_names_kitti)

img_names_nus = os.listdir(data_root_nus)
img_names_nus = natsort.natsorted(img_names_nus)

img_names_waymo = os.listdir(data_root_waymo)
img_names_waymo = natsort.natsorted(img_names_waymo)


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
              pos=(0, 0),
              font_scale=1,
              font_thickness=1,
              text_color=(255, 255, 255),
              text_color_bg=(255, 255, 255)
              ):
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    pos = (int(pos[0] - text_w / 2), pos[1] - text_h)
    x, y = pos
    # cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale,
                text_color, font_thickness)

    return text_size


semsegcolors = cv2.imread(
    '/home/gopi/OneDrive/Papers/RetSeg3D/semsegcolors.drawio_1.png')
nussegcolors = cv2.imread(
    '/home/gopi/OneDrive/Papers/RetSeg3D/nuscolors.drawio.png')
waymosegcolors = cv2.imread(
    '/home/gopi/OneDrive/Papers/RetSeg3D/waymosegcolors.drawio.png')
semsegcolors2 = cv2.imread('/home/gopi/OneDrive/Papers/RetSeg3D'
                           '/semsegcolors-Page-2.drawio.png')

# VIDEO
video_path_kitti = ('/home/gopi/PhD/workdirs/Results/RetSeg3D/semantickitti'
                    '/val/predictions.mp4')
video_path_nus = ('/home/gopi/PhD/workdirs/Results/RetSeg3D/nuscenes/val'
                  '/predictions.mp4')
video_path_waymo = ('/home/gopi/PhD/workdirs/Results/RetSeg3D/waymo/val'
                    '/predictions.mp4')
out = cv2.VideoWriter(
    video_path_waymo,
    cv2.VideoWriter_fourcc(*'mp4v'),  # MJPG XVID MP4V
    18, (1848, 1016))

# # KITTI
# # count = 0
# for imgname in tqdm(img_names_kitti):
#     img_path = osp.join(data_root_kitti, imgname)
#     img = cv2.imread(img_path)  # (1848, 1016)
#
#     # kitti
#     draw_text(img, 'Semantic KITTI Dataset', pos=(924, 50),
#               font=cv2.FONT_HERSHEY_COMPLEX)
#     draw_text(img, 'Ground Truth', pos=(340, 100))
#     draw_text(img, 'Prediction', pos=(924, 100))
#     draw_text(img, 'Error', pos=(1530, 100))
#
#     # kitti
#     img[830:861, 350:875] = semsegcolors
#     img[845:860, 1470:1583] = semsegcolors2
#
#     # cv2.imshow('img', img)
#     # cv2.waitKey()
#
#     # img_array.append(img)
#     out.write(img)
#
#     # count += 1
#     # if count == 300:
#     #     break
#
# Waymo
count = 0
for imgname in tqdm(img_names_waymo):
    img_path = osp.join(data_root_waymo, imgname)
    img = cv2.imread(img_path)  # (1848, 1016)

    # nuscenes
    draw_text(img, 'Waymo Dataset', pos=(924, 50),
              font=cv2.FONT_HERSHEY_COMPLEX)
    draw_text(img, 'Ground Truth', pos=(340, 100))
    draw_text(img, 'Prediction', pos=(924, 100))
    draw_text(img, 'Error', pos=(1530, 100))

    # nuscenes
    img[830:861, 350:1034] = waymosegcolors
    img[845:860, 1470:1583] = semsegcolors2

    # cv2.imshow('img', img)
    # cv2.waitKey()

    # img_array.append(img)
    out.write(img)

    # count += 1
    # if count == 300:
    #     break

# # NuScenes
# # count = 0
# for imgname in tqdm(img_names_nus):
#     img_path = osp.join(data_root_nus, imgname)
#
#     # PIL TO CV2
#     # pil_image = Image.open(img_path).convert('RGB')
#     # open_cv_image = np.array(pil_image)
#     # # Convert RGB to BGR
#     # img = open_cv_image[:, :, ::-1].copy()
#
#     img = cv2.imread(img_path)  # (1848, 1016)
#
#     # nuscenes
#     draw_text(img, 'NuScenes Dataset', pos=(924, 50),
#               font=cv2.FONT_HERSHEY_COMPLEX)
#     draw_text(img, 'Ground Truth', pos=(340, 100))
#     draw_text(img, 'Prediction', pos=(924, 100))
#     draw_text(img, 'Error', pos=(1530, 100))
#
#     # nuscenes
#     img[830:859, 350:858] = nussegcolors
#     img[845:860, 1470:1583] = semsegcolors2
#
#     # cv2.imshow('img', img)
#     # cv2.waitKey()
#
#     # img_array.append(img)
#     out.write(img)
#
#     # count += 1
#     # if count == 300:
#     #     break
#


out.release()
