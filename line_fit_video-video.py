import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from Line import Line
from History import History
from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset, viz1
from moviepy.editor import VideoFileClip

# Global variables (just to make the moviepy video annotation work)


left_curve, right_curve = 0., 0.  # 左右车道的曲率半径
left_lane_inds, right_lane_inds = None, None  # 用于计算曲率
arrfitL = [] #存储左车道线的参数和曲率
arrfitR = [] #存储右车道线的参数和曲率

alpha = .70     # 指数平滑系数（值越大，离当前最近的数值权重越大）
top_x_thresh = 0.5  # 曲率最大变化阈值，超过阈值的滤除
bottom_x_thresh = 0.35    # x轴截距偏离历史帧的阈值，超过阈值的滤除
maxHistoryNum = 5  # 最大历史帧数
# global countFilter
# countFilter = 2   # 过滤次数
top_x_left_history = History(n=maxHistoryNum) #存储历史左车道线的曲率
top_x_right_history = History(n=maxHistoryNum) #存储历史右车道线的曲率
bottom_x_left_history = History(n=maxHistoryNum) #存储历史左车道线的x轴截距
bottom_x_right_history = History(n=maxHistoryNum) #存储历史右车道线的x轴截距

left_LaneFit_history = History(n=maxHistoryNum) #存储历史左车道线系数
right_LaneFit_history = History(n=maxHistoryNum) #存储历史右车道线系数

# with open('calibrate_camera.p', 'rb') as f:
#     save_dict = pickle.load(f)
# mtx = save_dict['mtx']
# dist = save_dict['dist'] # 矫正

# mtx = np.float32([[959.7910, 0, 696.0217],
#                   [0, 956.9251, 224.1806],
#                   [0, 0, 1]])
# dist = np.float32([-0.3691481, 0.1968681, 0.001353473, 0.0005677587, -0.06770705])

window_size = 5  # 线条平滑帧数
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False  # 快速线拟合检测线？
# left_curve, right_curve = 0., 0.  # 左右车道的曲率半径
# left_lane_inds, right_lane_inds = None, None  # 用于计算曲率

# MoviePy视频注释将调用此函数
def annotate_image(img_in):
    """
    使用车道线标记标注输入图像
    返回带注释的图像
    """
    detectedfast = False  # 根据上一帧结果拟合检测线，省略每次求直方图
    T = 0.8


    global mtx, dist, left_line, right_line, detected, arrfitL, arrfitR
    global left_curve, right_curve, left_lane_inds, right_lane_inds
    global top_x_left_history, top_x_right_history, bottom_x_left_history, bottom_x_right_history
    global left_LaneFit_history, right_LaneFit_history
    # Undistort未失真, threshold阈值, perspective transform视角转换
    # undist = cv2.undistort(img_in, mtx, dist, None, mtx)
    undist = img_in
    # 边缘检测阈值
    img, abs_bin, mag_bin, dir_bin, WY_BIN = combined_thresh(undist)
    # 透视变换
    binary_warped, binary_unwarped, m, m_inv = perspective_transform(img)

    # cv2.imshow('img', undist)
    # cv2.imshow('binary_warped', binary_warped)
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
            out_img = ret['out_img']
            histo = ret['histo']

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
    vehicle_offset, bottom_x_left, bottom_x_right, top_x_left, top_x_right = calc_vehicle_offset(undist, left_fit, right_fit)

    """
    PMH：采用历史帧更新数据
    """
    top_x_left_history, bottom_x_left_history, left_LaneFit_history, is_left_filtered = filter_history(top_x_left_history, bottom_x_left_history, left_LaneFit_history,
                                                                                                       top_x_left, bottom_x_left, left_fit)
    top_x_right_history, bottom_x_right_history, right_LaneFit_history, is_right_filtered = filter_history(top_x_right_history, bottom_x_right_history, right_LaneFit_history,
                                                                                                           top_x_right, bottom_x_right, right_fit)


    if is_left_filtered is True:
        print('左车道线被过滤！')
        left_fit = left_LaneFit_history.get_latest()
    else:
        left_fit = left_LaneFit_history.get_smoothed()

    if is_right_filtered is True:
        print('右车道线被过滤！')
        right_fit = right_LaneFit_history.get_latest()
    else:
        right_fit = right_LaneFit_history.get_smoothed()

    # # 存储左车道线(曲线的参数a,b,c、曲率半径、车道偏移量、图片序号)
    # tupcurveL = (left_curve, vehicle_offset, bottom_x_left, num_img)
    # tupcurveL = (left_curve, bottom_x_left)
    # newleft = left_fit + tupcurveL
    # arrfitL.append(newleft)
    #
    # # 存储右车道线(曲线的参数a,b,c、曲率半径、车道偏移量、图片序号)
    # # tupcurveR = (right_curve, vehicle_offset, bottom_x_right, num_img)
    # tupcurveR = (right_curve, bottom_x_right)
    # newright = right_fit + tupcurveR
    # arrfitR.append(newright)

    # Perform final visualization on top of original undistorted image

    result, color_warp, new_warp, newwarpNO = final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset, is_left_filtered, is_right_filtered)
    # cv2.imshow('result', result)
    # viz1(binary_warped, ret, save_file=None)

    # return img, WY_BIN, binary_unwarped, binary_warped, color_warp, result, arrfitL, arrfitR, new_warp, newwarpNO, out_img, histo
    # return arrfitL, arrfitR, new_warp, newwarpNO, color_warp, result, out_img, histo
    return result


def filter_history(top_x_history, bottom_x_history, lane_fit_history, new_top_x, new_bottom_x, new_lane_fit):
    # global countFilter
    if len(top_x_history.data) is 0:
        print('第一次记录历史信息')
        update = True
    else:
        # 上一帧信息
        last_top_x = top_x_history.get_latest()
        # last_bottom_x = bottom_x_left_history.get_mid()    #获得前5帧的中值，使用same_padding,往前面补4个相同数
        last_bottom_x = bottom_x_history.get_latest()
        last_lane_fit = lane_fit_history.get_latest()
        # 如果斜率、x轴截距超过指定阈值，当前车道线用上一帧结果代替
        a = abs((last_top_x - new_top_x) / last_top_x)
        b = abs((last_bottom_x - new_bottom_x) / last_bottom_x)
        # print('上截距变化率a = %05f        下截距变化率b = %05f'%(a,b))
        if (a > top_x_thresh) \
                or (b > bottom_x_thresh):
            # print('结果相对历史预测跳动过大！')
            update = False
            # countFilter += 1
            # if countFilter >= 7:
            #     update = True
            #     print('连续跳动过大！自动更新结果')
        else:
            update = True
            # countFilter = 0

    if update is True:
        # 更新历史车道线信息
        top_x_history = top_x_history.update_history(new_top_x)  # 存储新车道线的曲率
        bottom_x_history = bottom_x_history.update_history(new_bottom_x)  # 存储新车道线的x轴截距
        lane_fit_history = lane_fit_history.update_history(new_lane_fit)  # 存储新车道线的系数信息
        is_filtered = False
    else:
        top_x_history = top_x_history.update_history(last_top_x)  # 存储历史车道线的曲率
        bottom_x_history = bottom_x_history.update_history(last_bottom_x)  # 存储历史车道线的x轴截距
        lane_fit_history = lane_fit_history.update_history(last_lane_fit)  # 存储历史车道线的拟合系数
        is_filtered = True

    """
    # 更新截距
    """
    # bottom_x_history = bottom_x_history.update_history(new_bottom_x)  # 存储新车道线的x轴截距


    # 指数平滑历史车道线信息
    top_x_history = top_x_history.add_smoothing(alpha)
    bottom_x_history = bottom_x_history.add_smoothing(alpha)
    lane_fit_history = lane_fit_history.smooth_coeffs(alpha)

    return top_x_history, bottom_x_history, lane_fit_history, is_filtered



def annotate_video(input_file, output_file):
    """ Given input_file video, save annotated video to output_file """
    # video = VideoFileClip(input_file)
    # annotated_video = video.fl_image(annotate_image)
    # annotated_video.write_videofile(output_file, audio=False)
    # fourcc = cv2.CV_FOURCC('M', 'J', 'P', 'G')

    fps = 8  # 视频帧率
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    videoWriter = cv2.VideoWriter(output_file, fourcc, fps, (1920, 1080))  # (1360,480)为视频大小
    for i in range(18660, 19300):
        img_file = '%s%06d%s' % (input_file, i, 'L.jpg')
        print(img_file)
        img12 = cv2.imread(img_file)
        converted = cv2.cvtColor(img12, cv2.COLOR_BGR2RGB)
        result_RGB = annotate_image(converted)
        result = cv2.cvtColor(result_RGB, cv2.COLOR_RGB2BGR)
        # cv2.imshow('img', result)
        # cv2.waitKey(1000/int(fps))
        videoWriter.write(result)
    videoWriter.release()


if __name__ == '__main__':
	# Annotate the video

    # videodir = 'D:/CIDI/data/cidi20180505_highway_lane_video/20180505133013577.wmv'
    output_file = 'D:/CIDI/data/L/18670.avi'
    imagedir = 'D:/CIDI/data/L/Rectified_L/'
    annotate_video(imagedir, output_file)


"""
if __name__ == '__main__':
    #透视变换矩阵
    hightdis = [360, 1002]  # dst矩阵上下高度
    widthdis = [550, 1300]  # dst矩阵左右宽度
    src = np.float32(
        [[200, 950],  # 左下
         [1820, 950],  # 右下
         [720, 430],  # 左上
         [1250, 430]])  # 右上

    dst = np.float32(
        [[widthdis[0], hightdis[1]],
         [widthdis[1], hightdis[1]],
         [widthdis[0], hightdis[0]],
         [widthdis[1], hightdis[0]]])

    x = [src[0, 0], src[1, 0], src[3, 0], src[2, 0], src[0, 0]]
    y = [src[0, 1], src[1, 1], src[3, 1], src[2, 1], src[0, 1]]

    plot_sub = True # 画图设置， 如果为True则画出6个子图

    detectedfast = False # 控制detected

    savefit = True # 是否保存二次函数参数、曲率、车道偏移量

    saveimg = True  # 如果为Trlue则保存输出图，否则只显示图片

    savenewward = False # 保存行驶区域
    # 获取图像下半部分的直方图，阈值T = 0.5 ，T越大获取的图像部分越大
    T = 0.8

    filedir = 'D:/CIDI/data/cidi20180505_highway/L/image_201805051108L/'
    imgdir = filedir + '06000-11999/'
    savedir = filedir + '06000-11999-result4/'
    newwarddir = filedir + 'lane_ROI/'  # 存储行驶区域模板
    # newwarddir 存储行驶区域模板
    imgformat = 'L.jpg'

    plt.figure(1)
    # 图片序号i：#cidi20180418：1-390 709:790
    for i in range(9600, 11999):
        print(detected)
        orig = '%s%06d' % ('orig:', i)
        # print(orig)
        img_file = '%s%06d%s' % (imgdir, i, imgformat)
        print(img_file)
        try:
            img1 = mpimg.imread(img_file)
        except:
            continue

        print(img1.shape)
        # plt.imshow(img1)
        # plt.show()
        arrfitL, arrfitR, new_warp, newwarpNO, color_warp, result, out_img, histo = annotate_image(img1, src, dst, detectedfast, T, i)
        # img, WY_BIN, binary_unwarped, binary_warped, color_warp, result, arrfitL, arrfitR, new_warp, newwarpNO, out_img, histo = annotate_image(img1, src, dst, detectedfast, T, i)
        # result, new_warp, newwarpNO = annotate_image(img1, src, dst, detectedfast, T, i)
        # --------------画图-----------------
        # plt.figure(1)

        if plot_sub:
            fig = plt.gcf()
            fig.set_size_inches(16.5, 8)
            # # 边缘图像
            # plt.subplot(2, 3, 1)
            # plt.imshow(img, cmap='gray', vmin=0, vmax=1)
            # plt.plot(x, y)
            # plt.title('histo')
            # # 未变换的ROI
            # plt.subplot(2, 3, 2)
            # plt.imshow(WY_BIN, cmap='gray', vmin=0, vmax=1)
            # # plt.imshow(binary_unwarped, cmap='gray', vmin=0, vmax=1)
            # plt.plot(x, y)
            # plt.title('ROI')
            # # 变换后的ROI
            # plt.subplot(2, 3, 3)
            # plt.imshow(binary_warped, cmap='gray', vmin=0, vmax=1)
            # plt.title('warped ROI')
            # # 结果图
            # plt.subplot(2, 3, 4)
            # plt.imshow(result)
            # plt.title('result')
            # plt.plot(x, y)
            # # 滑移窗口
            # plt.subplot(2, 3, 5)
            # plt.imshow(out_img)
            # plt.plot(x, y)
            # plt.title('Search Process')
            # # 变换后的拟合线
            # plt.subplot(2, 3, 6)
            # plt.imshow(color_warp, cmap='gray', vmin=0, vmax=1)
            # plt.title('fit line')

            # 直方图
            plt.subplot(2, 2, 1)
            plt.plot(histo)
            plt.title('histo')
            # 结果图
            plt.subplot(2, 2, 2)
            plt.imshow(result)
            plt.title('result')
            plt.plot(x, y)
            # 滑移窗口
            plt.subplot(2, 2, 3)
            plt.imshow(out_img)
            plt.plot(x, y)
            plt.title('Search Process')
            # 变换后的拟合线
            plt.subplot(2, 2, 4)
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
            fig.set_size_inches(20,10)
            plt.savefig(save_file)
        else:
            plt.show()

        if savenewward:
            save_ward = '%s%06d%s' % (newwarddir, i, 'Lr.jpg')
            cv2.imwrite(save_ward,new_warp)
            save_wardNO = '%s%06d%s' % (newwarddir, i, 'LrNO.jpg')
            cv2.imwrite(save_wardNO, newwarpNO)

        plt.clf() # 清图。

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
"""
