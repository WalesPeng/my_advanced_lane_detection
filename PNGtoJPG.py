import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image as PI

for i in range(543, 641):
    img_file = '%s%06d%s' % ('D:/CIDI/data/LINE/Rectified_L/', i, 'L.png')
    img = PI.open(img_file)
    print(img_file)
    img1 = mpimg.imread(img_file)
    print(img1.shape)
    save_file = '%s%06d%s' % ('D:/CIDI/data/LINE/JPG/', i, 'L.jpg')
    img.save(save_file)