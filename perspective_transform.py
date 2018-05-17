import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh

def perspective_transform(img):
    #     """
    #     Execute perspective transform
    #     """
    img_size = (img.shape[1], img.shape[0])

    hightdis = [300, 1002]  # dst矩阵上下高度
    widthdis = [450, 1300]  # dst矩阵左右宽度
    src = np.float32(
        [[200, 1050],  # 左下
         [1820, 1050],  # 右下
         [720, 530],  # 左上
         [1250, 530]])  # 右上

    dst = np.float32(
        [[widthdis[0], hightdis[1]],
         [widthdis[1], hightdis[1]],
         [widthdis[0], hightdis[0]],
         [widthdis[1], hightdis[0]]])

    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG

    return warped, unwarped, m, m_inv


# def perspective_transform(img, src, dst):
#     """
#     Execute perspective transform
#     """
#     img_size = (img.shape[1], img.shape[0])
#
#     m = cv2.getPerspectiveTransform(src, dst)
#     m_inv = cv2.getPerspectiveTransform(dst, src)
#
#     warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
#     unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
#
#     return warped, unwarped, m, m_inv


if __name__ == '__main__':
    img_file = 'test_images/KITTI/0000000195.jpg'
    # img_file = 'test_images/test1.jpg'

    # with open('calibrate_camera.p', 'rb') as f:
    #     save_dict = pickle.load(f)
    # mtx = save_dict['mtx']
    # dist = save_dict['dist']

    mtx = np.float32([[959.7910, 0, 696.0217],
           [0, 956.9251, 224.1806],
           [0, 0, 1]])
    dist = np.float32([-0.3691481, 0.1968681, 0.001353473, 0.0005677587, -0.06770705])

    img = mpimg.imread(img_file)
    img = cv2.undistort(img, mtx, dist, None, mtx)
    # 边缘检测
    img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)

    warped, unwarped, m, m_inv = perspective_transform(img)

    plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
    plt.show()

    plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)
    plt.show()
