import os
import cv2
import glob
import numpy as np

def to_video(folder_path, out_path, fps=4, downsample=1, caption_frame=None): #cby
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    imgs_path = glob.glob(os.path.join(folder_path, '*.jpg'))
    imgs_path = sorted(imgs_path, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    img_array = []
    caption_height = 66 * 34
    img_id_pre = -2
    img_id_start = -2
    caption_id = -2
    video_count = 0
    for img_path in imgs_path:
        img_id = int(img_path.split('/')[-1].split('.')[0])
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        is_null = np.all(img[height-caption_height:height, :, :] == 255)
        if img_id - img_id_pre != 1:
            img_id_start = img_id
            if video_count != 0:
                out = cv2.VideoWriter(
                    out_path.replace('*', str(video_count)), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
                for i in range(len(img_array)):
                    out.write(img_array[i])
                out.release()
                img_array = []
            video_count += 1
            caption_img = img[height-caption_height:height, :, :]
            if is_null:
                caption_id = -1
            else:
                caption_id = img_id
        elif caption_id==-1 or (img_id - caption_id)>=caption_frame:
            if not is_null:
                caption_img = img[height-caption_height:height, :, :]
                caption_id = img_id
            else:
                img[height-caption_height:height, :, :] = caption_img
        else:
            img[height-caption_height:height, :, :] = caption_img
        print(img_id, img_id_pre, img_id_start, caption_id, is_null)
        img_id_pre = img_id
        img = cv2.resize(img, (width//downsample, height //
                            downsample), interpolation=cv2.INTER_AREA)
        height_2, width_2, channel = img.shape
        size = (width_2, height_2)
        img_array.append(img)

    
to_video('test/base_caption/Mon_Jun_10_15_58_16_2024/vis_output',
         'test/base_caption/Mon_Jun_10_15_58_16_2024/test_demo/test_demo*.avi',
         fps=4, downsample=2, caption_frame=12)