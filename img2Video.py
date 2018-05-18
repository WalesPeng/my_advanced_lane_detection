import numpy as np
import cv2
import os

def Image2Video(img_root, fps, video_name):
    img_names = os.listdir(img_root)
    starti = 0;
    endi =len(img_names)  #100

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    for i in range(starti, endi):
        try:
            img_file = img_root + img_names[i]
            frame = cv2.imread(img_file)
            print(img_file)
        except:
            continue
        if i == starti:
            videoWriter = cv2.VideoWriter(video_name, fourcc, fps, (frame.shape[1], frame.shape[0]))  # 最后一个是保存图片的尺寸
        videoWriter.write(frame)

    videoWriter.release()
    return

if __name__ == '__main__':
    # img_root = 'D:/data/cidi20180505_highway/L/image_201805051108L/0000-5999/'
    img_root = 'D:/data/apollo_result_pic/'
    # Image2Video(img_root, 24, 'saveVideo.avi')
    Image2Video(img_root, 8, 'saveVideo.avi')