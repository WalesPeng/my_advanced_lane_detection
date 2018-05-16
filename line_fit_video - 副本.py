import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from Line import Line
from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset, viz1
from moviepy.editor import VideoFileClip

# Global variables (just to make the moviepy video annotation work)
window_size = 5  # 线条平滑帧数
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False  # 快速线拟合检测线？

left_curve, right_curve = 0., 0.  # 左右车道的曲率半径
left_lane_inds, right_lane_inds = None, None  # 用于计算曲率
arrfitL = [] #存储左车道线的参数和曲率
arrfitR = [] #存储右车道线的参数和曲率
# with open('calibrate_camera.p', 'rb') as f:
#     save_dict = pickle.load(f)
# mtx = save_dict['mtx']
# dist = save_dict['dist'] # 矫正

# mtx = np.float32([[959.7910, 0, 696.0217],
#                   [0, 956.9251, 224.1806],
#                   [0, 0, 1]])
# dist = np.float32([-0.3691481, 0.1968681, 0.001353473, 0.0005677587, -0.06770705])

# window_size = 5  # 线条平滑帧数
# left_line = Line(n=window_size)
# right_line = Line(n=window_size)
# # detected = False  # 快速线拟合检测线？
# left_curve, right_curve = 0., 0.  # 左右车道的曲率半径
# left_lane_inds, right_lane_inds = None, None  # 用于计算曲率

# MoviePy视频注释将调用此函数
def annotate_image(img_in, src, dst, detectedfast, T, num_img):
    """
    使用车道线标记标注输入图像
    返回带注释的图像
    """

    global mtx, dist, left_line, right_line, detected, arrfitL, arrfitR
    global left_curve, right_curve, left_lane_inds, right_lane_inds
    # Undistort未失真, threshold阈值, perspective transform视角转换
    # undist = cv2.undistort(img_in, mtx, dist, None, mtx)
    undist = img_in
    # 边缘检测阈值
    img, abs_bin, mag_bin, dir_bin, WY_BIN= combined_thresh(undist)
    # 透视变换
    binary_warped, binary_unwarped, m, m_inv = perspective_transform(img, src, dst)
    # Perform polynomial fit执行多项式拟合
    if not detected:
        # Slow line fit
        ret = line_fit(binary_warped, T)
        if len(ret) > 0 :
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

            # Get moving average of line fit coefficients
            # 获得线性拟合系数的移动平均值
            left_fit = left_line.add_fit(left_fit)
            right_fit = right_line.add_fit(right_fit)

            # Calculate curvature计算曲率
            left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
            if detectedfast:
              detected = True  # slow line fit always detects the line
        else:
            return img_in
    else:  # implies detected == True
        # Fast line fit
        left_fit = left_line.get_fit()
        right_fit = right_line.get_fit()
        ret = tune_fit(binary_warped, left_fit, right_fit)
        left_fit = ret['left_fit']
        right_fit = ret['right_fit']
        nonzerox = ret['nonzerox']
        nonzeroy = ret['nonzeroy']
        left_lane_inds = ret['left_lane_inds']
        right_lane_inds = ret['right_lane_inds']

        # Only make updates if we detected lines in current frame
        if ret is not None:
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

            left_fit = left_line.add_fit(left_fit)
            right_fit = right_line.add_fit(right_fit)
            left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
        else:
            detected = False
    # 偏移量
    vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)
    # 存储左车道线(曲线的参数a,b,c、曲率半径、车道偏移量、图片序号)
    tupcurveL = (left_curve, vehicle_offset, num_img)
    newleft = left_fit + tupcurveL
    arrfitL.append(newleft)
    # 存储右车道线(曲线的参数a,b,c、曲率半径、车道偏移量、图片序号)
    tupcurveR = (right_curve, vehicle_offset, num_img)
    newright = right_fit + tupcurveR
    arrfitR.append(newright)
    # Perform final visualization on top of original undistorted image
    result, color_warp, new_warp, newwarpNO = final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset)

    # viz1(binary_warped, ret, save_file=None)

    return img, binary_unwarped, binary_warped, color_warp, result, arrfitL, arrfitR, new_warp, newwarpNO
    # return result

def process_image(img):
    print(img.shape)
        # plt.imshow()
        # plt.show()
    img, binary_unwarped, binary_warped, color_warp, result, arrfitL, arrfitR, new_warp, newwarpNO = annotate_image(
            img1, src, dst, detectedfast, T, i)
        # result = annotate_image(img1, src, dst, detectedfast, T, i)
        # --------------画图-----------------
    plt.figure(1)
    if plot_sub:
            # 边缘图像
        plt.subplot(2, 3, 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.plot(x, y)
        plt.title('sobel')
            # 未变换的ROI
        plt.subplot(2, 3, 2)
        plt.imshow(binary_unwarped, cmap='gray', vmin=0, vmax=1)
        plt.plot(x, y)
        plt.title('ROI')
        # 变换后的ROI
        plt.subplot(2, 3, 3)
        plt.imshow(binary_warped, cmap='gray', vmin=0, vmax=1)
        plt.title('warped ROI')
        # 原图
        plt.subplot(2, 3, 4)
        plt.imshow(img1)
        plt.plot(x, y)
        plt.title(orig)
        # 结果图
        plt.subplot(2, 3, 5)
        plt.imshow(result)
        plt.title('result')
        plt.plot(x, y)
        # 变换后的拟合线
        plt.subplot(2, 3, 6)
        plt.imshow(color_warp, cmap='gray', vmin=0, vmax=1)
        plt.title('fit line')

        # plt.tight_layout()

    else:
        plt.imshow(result)
        plt.plot(x, y)
        plt.title(orig)

    if saveimg:
        save_file = '%s%06d%s' % (savedir, i, imgformat)
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.savefig(save_file)
    else:
        plt.show()

    if savenewward:
        save_ward = '%s%06d%s' % (newwarddir, i, 'Lr.jpg')
        cv2.imwrite(save_ward, new_warp)
        save_wardNO = '%s%06d%s' % (newwarddir, i, 'LrNO.jpg')
        cv2.imwrite(save_wardNO, newwarpNO)


if __name__ == '__main__':
#透视变换矩阵
    hightdis = [490, 815] #dst矩阵上下高度
    widthdis = [437, 1100] # dst矩阵左右宽度
    src = np.float32(
        [[407, 841], #左下
        [1736, 840], #右下
        [644, 551], #左上
        [1394, 551]]) #右上
    dst = np.float32(
        [[widthdis[0], hightdis[1]],
        [widthdis[1], hightdis[1]],
        [widthdis[0], hightdis[0]],
        [widthdis[1], hightdis[0]]])

    x = [src[0, 0], src[1, 0], src[3, 0], src[2, 0], src[0, 0]]
    y = [src[0, 1], src[1, 1], src[3, 1], src[2, 1], src[0, 1]]

    plot_sub = True # 画图设置， 如果为True则画出6个子图

    detectedfast = False # 控制detected

    savefit = False # 是否保存二次函数参数、曲率、车道偏移量

    saveimg = False  # 如果为Trlue则保存输出图，否则只显示图片

    savenewward = False # 保存行驶区域
    # 获取图像下半部分的直方图，阈值T = 0.5 ，T越大获取的图像部分越大
    T = 0.8

    # 图片路径和存储路径
    filedir = 'D:/CIDI/data/LINE/'
    imgdir = filedir + 'JPG/'
    savedir = filedir + 'result1/'
    newwarddir = filedir + 'newward1/'  # 存储行驶区域模板
    # newwarddir 存储行驶区域模板
    imgformat = 'L.jpg'
    # 图片序号i：#cidi20180418：1-390 709:790
    for i in range(343, 641):
        print(detected)
        orig = '%s%06d' % ('orig:', i)
        # print(orig)
        img_file = '%s%06d%s' % (imgdir, i, imgformat)
        print(img_file)
        try:
            img1 = mpimg.imread(img_file)
        except:
            continue


    if savefit:
        filename1 = '%s%s' % (savedir,'arrfitL.txt')
        f1 = open(filename1, 'w')
        for i in arrfitL:
            f1.write(str(i))
            f1.write("\n")
        f1.close()

        filename2 = '%s%s' % (savedir,'arrfitR.txt')
        f2 = open(filename2, 'w')
        for i in arrfitR:
            f2.write(str(i))
            f2.write("\n")
        f2.close()
