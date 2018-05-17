import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
    """
    Takes an image, gradient orientation, and threshold min/max values
    给定一个图像，梯度方向和阈值最小值/最大值
    """
    if img.shape.__len__() == 3:
    # Convert to grayscale转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
    """
    Return the magnitude of the gradient
    for a given sobel kernel size and threshold values
    返回给定的sobel内核大小和阈值的梯度大小
    """
    if img.shape.__len__() == 3:
    # Convert to grayscale转换为灰度    # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Return the direction of the gradient
    for a given sobel kernel size and threshold values
    返回给定sobel内核大小和阈值的梯度方向
    """
    if img.shape.__len__() == 3:
    # Convert to grayscale转换为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def hls_thresh(img, thresh=(100, 255)):
    """
    Convert RGB to HLS and threshold to binary image using S channel
    使用S通道将RGB转换为HLS并将阈值转换为二进制图像
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

# 在HSV空间下进行白色和黄色的筛选
def select_white_yellow(image):
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # white color mask
    lower = np.uint8([0, 0, 150])
    upper = np.uint8([50, 50, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower1 = np.uint8([10,  80, 210])
    upper1 = np.uint8([30, 110, 255])
    yellow_mask = cv2.inRange(converted, lower1, upper1)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    binary_output = np.zeros_like(white_mask)
    binary_show = np.zeros_like(white_mask)
    binary_show = cv2.bitwise_and(image, image, dst=None, mask=mask)
    binary_output[mask>0] = 1

    # cv2.imshow('binary_output', binary_show)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # binary_output_opened = cv2.morphologyEx(binary_output, cv2.MORPH_OPEN, kernel)
    # binary_output = cv2.dilate(binary_output_opened, kernel)
    # plt.figure(6)
    # plt.imshow(binary_output, cmap='gray', vmin=0, vmax=1)

    return binary_output

# 在RGB空间下对白色筛选和HSV空间下对黄色筛选
def filter_colors(image, white_threshold = 150):
    """
    Filter the image to include only yellow and white pixels
    只保留白线和黄线的像素
    """
    # Filter white pixels:white_threshold
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([255, 255, 255])

    white_mask = cv2.inRange(image, lower_white, upper_white)


    # white_image = cv2.bitwise_and(image, image, mask=white_mask)

    # Filter yellow pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([90, 100, 100])
    upper_yellow = np.array([110, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
    binary_output = np.zeros_like(white_mask)
    binary_output[white_mask==255] = 1
    binary_output[yellow_mask == 255] = 1
    # # Combine the two above images
    # image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

    return binary_output

def combined_thresh(img):
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.GaussianBlur(blur, (5, 5), 0)
    RGB_bin = filter_colors(img, white_threshold = 150)
    # plt.figure(4)
    # plt.imshow(RGB_bin)
    abs_bin = abs_sobel_thresh(img, orient='x', thresh_min=35, thresh_max=255)
    mag_bin = mag_thresh(img, sobel_kernel=3, mag_thresh=(35, 150))
    dir_bin = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.4)) #57.3*0.7=40.13 1.3*57.3=74
    combined = np.zeros_like(dir_bin)
    print(img.shape.__len__())
    if img.shape.__len__() == 3:
    # Convert to grayscale转换为灰度
        hls_bin = hls_thresh(img, thresh=(100, 255))
        WY_BIN = select_white_yellow(img)
        # mask = ((abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))) | hls_bin == 1) & RGB_bin == 1
        mask = ((abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1)))) & WY_BIN == 1
        # mask = ((abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1)))) == 1

    else:
        mask = abs_bin == 1 | ((mag_bin == 1) & (dir_bin == 1))

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    combined[mask] = 1

    return combined, abs_bin, mag_bin, dir_bin, WY_BIN  # DEBUG


if __name__ == '__main__':
    i = 8
    orig = '%s%06d' % ('orig:', i)
    filedir = 'D:/CIDI/data/LINE/'
    imgdir = filedir + 'JPG/'
    imgformat = 'L.jpg'
    img_file = '%s%06d%s' % (imgdir, i, imgformat)
    # imgdir = 'D:/data/lane_line/cidi20180418/PNG-路段1-不同目标测距/JPG1/'
    # img_file = '%s%06d%s' % (imgdir, i, 'Lr.jpg')
    # img_file = 'D:/data/CIDI/highway/img/875_3300.jpg'
    # imgdir = 'D:/data/lane_line/2011_09_26_drive_0015_sync/KITTI/'
    # img_file = '%s%010d%s' % (imgdir, i, '.jpg')
    # imgdir = 'D:/data/PNG20180206dataAllRectJPG/JPG/'
    # img_file = '%s%06d%s' % (imgdir, i, 'Lr.jpg')


    # with open('calibrate_camera.p', 'rb') as f:
    #     save_dict = pickle.load(f)
    # mtx = save_dict['mtx'] #内参矩阵
    # dist = save_dict['dist'] #畸变参数

    # mtx = np.float32([[959.7910, 0, 696.0217],
    #        [0, 956.9251, 224.1806],
    #        [0, 0, 1]])
    # dist = np.float32([-0.3691481, 0.1968681, 0.001353473, 0.0005677587, -0.06770705])

    # print(save_dict)
    img = mpimg.imread(img_file)
    # img = cv2.undistort(img, mtx, dist, None, mtx)

    combined, abs_bin, mag_bin, dir_bin, WY_BIN = combined_thresh(img)
    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.imshow(abs_bin, cmap='gray', vmin=0, vmax=1)
    plt.title('abs_bin')
    plt.subplot(2, 3, 2)
    plt.imshow(mag_bin, cmap='gray', vmin=0, vmax=1)
    plt.title('mag_bin')
    plt.subplot(2, 3, 3)
    plt.imshow(dir_bin, cmap='gray', vmin=0, vmax=1)
    plt.title('dir_bin')
    plt.subplot(2, 3, 4)
    plt.imshow(WY_BIN, cmap='gray', vmin=0, vmax=1)
    plt.title('WY_BIN')
    plt.subplot(2, 3, 5)
    plt.imshow(img)
    plt.title(orig)
    plt.subplot(2, 3, 6)
    plt.imshow(combined, cmap='gray', vmin=0, vmax=1)
    plt.title('combined')

    plt.tight_layout()
    plt.show()
