import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform


def line_fit(binary_warped, T):
    """
    Find and fit lane lines
    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    # 假设你已经创建了一个名为“binary_warped”的变形二进制图像，获取图像下半部分的直方图
    # axis=0 按列计算
    img_roi_y = 700  # [1]设置ROI区域的左上角的起点
    img_roi_x = 0
    img_roi_height = binary_warped.shape[0]  # [2]设置ROI区域的高度
    img_roi_width = binary_warped.shape[1]  # [3]设置ROI区域的宽度

    img_roi = binary_warped[img_roi_y:img_roi_height, img_roi_x:img_roi_width]
    # cv2.imshow('img_roi', img_roi)
    histogram = np.sum(img_roi[0 :, :], axis=0)
    # histogram = np.sum(img_roi[int(np.floor(binary_warped.shape[0]*(1-T))):,:], axis=0)
    # plt.show()
    # Create an output image to draw on and visualize the result
    # 创建一个输出图像来绘制并可视化结果
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
    cv2.rectangle(out_img, (img_roi_x, img_roi_y), (img_roi_width, img_roi_height), (255, 0, 0), 5)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    # 找出直方图左右两半的峰值 这些将成为左右线的起点
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[100:midpoint]) + 100
    rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint

    # PMH：如果一边未检测到车道线，即无直方图峰值，则根据另一条车道线复制一个搜索起点
    if (leftx_base == 100):
        leftx_base = np.argmax(histogram[midpoint:-100]) - midpoint
    if (rightx_base == midpoint):
        rightx_base = np.argmax(histogram[100:midpoint]) + midpoint
    # Choose the number of sliding windows 选择滑动窗口的数量
    nwindows = 9
    # Set height of windows
    # 设置窗口的高度 128
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    # 确定图像中所有非零像素的x和y位置
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    # 为每个窗口更新当前位置
    leftx_current = leftx_base
    rightx_current = rightx_base
    leftx_current_last = leftx_base
    rightx_current_last = rightx_base
    leftx_current_next = leftx_base
    rightx_current_next = rightx_base
    # Set the width of the windows +/- margin
    # 设置窗口+/-边距的宽度
    margin = 150
    # Set minimum number of pixels found to recenter window
    # 设置发现到最近窗口的最小像素数
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    # 创建空列表以接收左右车道像素索引
    left_lane_inds = []
    right_lane_inds = []

    # plt.figure(2)
    # plt.subplot(2, 1, 1)
    # plt.plot(histogram)
    # Step through the windows one by one
    # 逐一浏览窗口
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        # 确定x和y（以及右和左）的窗口边界
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        leftx_current = leftx_current_next
        rightx_current = rightx_current_next
        # 设置滑移窗口左右边界
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # 在可视化图像上绘制窗口
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)


        # plt.subplot(2, 1, 2)
        # plt.imshow(out_img, cmap='gray', vmin=0, vmax=1)
        # Identify the nonzero pixels in x and y within the window
        # 确定窗口内x和y的非零像素
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        # 将这些索引附加到列表中
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        # 如果找到> minpix像素，请在其平均位置上重新调整下一个窗口

        if len(good_left_inds) > minpix:
            leftx_current_next = np.int(np.mean(nonzerox[good_left_inds]))
        else:
            if window > 2:
                leftx_current_next = leftx_current + (leftx_current - leftx_current_last)
                # left_lane_inds.append(binary_warped.)     # 20180516 pmh 加入方框中点作为拟合点
            else:
                leftx_current_next = leftx_base

        if len(good_right_inds) > minpix:
            rightx_current_next = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            if window > 2:
                rightx_current_next = rightx_current + (rightx_current - rightx_current_last)
                # right_lane_inds.append(good_right_inds)
            else:
                rightx_current_next = rightx_base

        leftx_current_last = leftx_current
        rightx_current_last = rightx_current

    # plt.figure(2)
    # plt.subplot(2, 1, 1)
    # plt.plot(histogram)
    # plt.subplot(2, 1, 2)
    # plt.imshow(out_img, cmap='gray', vmin=0, vmax=1)

    # plt.savefig('D:/CIDI/data/L/line_fit_histo/')
    # plt.close()
    # save_file = '%s%06d%s' % ('D:/data/PNG20180206dataAllRectJPG/result1/', num_i+100000, 'Lr.jpg')
    # fig1 = plt.gcf()
    # fig1.set_size_inches(18.5, 10.5)
    # plt.savefig(save_file)
    # Concatenate the arrays of indices连接索引数组
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    # 提取左右线像素位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ret = {}
    # 如果车道线非空，则进行拟合二次曲线
    if (len(left_lane_inds) > 0) & (len(right_lane_inds) > 0):
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Return a dict of relevant variables
        ret['left_fit'] = left_fit
        ret['right_fit'] = right_fit
        ret['nonzerox'] = nonzerox
        ret['nonzeroy'] = nonzeroy
        ret['out_img'] = out_img
        ret['left_lane_inds'] = left_lane_inds
        ret['right_lane_inds'] = right_lane_inds
        ret['histo'] = histogram

    return ret


def tune_fit(binary_warped, left_fit, right_fit):
    """
    Given a previously fit line, quickly try to find the line based on previous lines
    给定一条先前合适的线条，快速尝试根据之前的线条找到线条
    """
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    # 假设你现在有一个来自下一帧视频的新的变形二进制图像（也称为“binary_warped”）现在找到线像素更容易了！
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    # 再次提取左右线像素位置
    leftx = nonzerox[left_lane_inds]      # 对一系列的bool变量返回 true 的 id 号
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # If we don't find enough relevant points, return all None (this means error)
    # 如果我们找不到足够的相关点，则返回全部无（这意味着错误）
    min_inds = 10
    if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
        return None

    # Fit a second order polynomial to each
    # 为每个拟合一个二阶多项式
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Return a dict of relevant variables
    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['left_lane_inds'] = left_lane_inds
    ret['right_lane_inds'] = right_lane_inds

    return ret


def viz1(binary_warped, ret, save_file=None):
    """
    Visualize each sliding window location and predicted lane lines, on binary warped image
    save_file is a string representing where to save the image (if None, then just display)
    在二值变形图像上显示每个滑动窗口位置和预测车道线save_file是一个字符串，表示图像的保存位置（如果为None，则仅显示）
    """
    # Grab variables from ret dictionary
    left_fit = ret['left_fit']
    right_fit = ret['right_fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    out_img = ret['out_img']
    left_lane_inds = ret['left_lane_inds']
    right_lane_inds = ret['right_lane_inds']

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
    plt.gcf().clear()


def viz2(binary_warped, ret, save_file=None):
    """
    Visualize the predicted lane lines with margin, on binary warped image
    save_file is a string representing where to save the image (if None, then just display)
    在二值变形图像上显示带边距的预测车道线save_file是表示图像保存位置的字符串（如果为None，则仅显示）
    """
    # Grab variables from ret dictionary
    left_fit = ret['left_fit']
    right_fit = ret['right_fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    left_lane_inds = ret['left_lane_inds']
    right_lane_inds = ret['right_lane_inds']

    # Create an image to draw on and an image to show the selection window
    out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    margin = 100  # NOTE: Keep this in sync with *_fit()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file)
    plt.gcf().clear()


def calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
    """
    Calculate radius of curvature in meters
    以米为单位计算曲率半径
    """
    # y_eval = 1160  # 720p video/image, so last (lowest on screen) y index is 719
    y_eval = 700   # 图像分辨率为1920*1080，取图像下1/3位置为计算斜率位置

    # Define conversions in x and y from pixels space to meters
    # 定义x和y从像素空间到米的转换5.86um
    ym_per_pix = 5.86/1000000 # meters per pixel in y dimensiony维度上每像素的米数30/720
    xm_per_pix = 5.86/1000000 # meters per pixel in x dimensiony维度上每像素的米数3.7/700

    # Extract left and right line pixel positions
    # 提取左右线像素位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials to x,y in world space
    # 将新的多项式拟合到世界空间中的x，y
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    # 计算新的曲率半径  等于曲率的倒数  曲率K = (|2a|) / (1 + (2ax + b)^2)^1.5
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / (2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / (2*right_fit_cr[0])
    # Now our radius of curvature is in meters现在我们的曲率半径以米为单位

    return left_curverad, right_curverad


def calc_vehicle_offset(undist, left_fit, right_fit):
    """
    Calculate vehicle offset from lane center, in meters
    计算车道中心的车辆偏移量，单位为米
    """
    # Calculate vehicle center offset in pixels
    top_y = 1
    bottom_y = undist.shape[0] - 1

    top_x_left = left_fit[0]*(top_y**2) + left_fit[1]*top_y + left_fit[2]
    top_x_right = right_fit[0]*(top_y**2) + right_fit[1]*top_y + right_fit[2]

    bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
    bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
    vehicle_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2

    # Convert pixel offset to meters
    xm_per_pix = 5.86/1000000 # meters per pixel in x dimension
    vehicle_offset *= xm_per_pix

    return vehicle_offset, bottom_x_left, bottom_x_right, top_x_left, top_x_right


def final_viz(undist, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset, is_left_filtered, is_right_filtered):
    """
    Final lane line prediction visualized and overlayed on top of original image
    最终车道线预测可视化并覆盖在原始图像的顶部
    """
    # Generate x and y values for plotting 为绘图生成x和y值
    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
    # 拟合的二次函数
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # 透过透视变换的矩阵求出点坐标（透视变换后的四个点----》原图的四个点）
    a1 = [left_fitx[0], ploty[0], 1]
    a2 = [left_fitx[undist.shape[0]-150], ploty[undist.shape[0]-150], 1]
    a3 = [right_fitx[0], ploty[0], 1]
    a4 = [right_fitx[undist.shape[0]-150], ploty[undist.shape[0]-150], 1]
    a = [a1, a2, a3, a4]
    rr1 = np.dot(a, m_inv.T)
    xx1 = np.ceil(rr1[:, 0] / rr1[:, 2]) # x坐标, 逆透视变换回来的坐标  np.ceil朝正无穷大取整数，注意除以缩放系数
    yy1 = np.ceil(rr1[:, 1] / rr1[:, 2]) # y坐标

    # 将车道线坐标点 经过逆透视变换后转换到原坐标系
    left_points = []
    right_points = []
    for i in range(len(left_fitx)):
        left_point = [left_fitx[i], ploty[i], 1]
        right_point = [right_fitx[i], ploty[i], 1]
        left_points.append(left_point)
        right_points.append(right_point)
    left_point_inv_trans = np.dot(left_points, m_inv.T)
    right_point_inv_trans = np.dot(right_points, m_inv.T)

    # 逆透视变换回来的车道线坐标，注意除以缩放系数。np.ceil朝正无穷大取整数。
    left_point_inv_xx = np.ceil(left_point_inv_trans[:, 0] / left_point_inv_trans[:, 2])
    left_point_inv_yy = np.ceil(left_point_inv_trans[:, 1] / left_point_inv_trans[:, 2])

    right_point_inv_xx = np.ceil(right_point_inv_trans[:, 0] / right_point_inv_trans[:, 2])
    right_point_inv_yy = np.ceil(right_point_inv_trans[:, 1] / right_point_inv_trans[:, 2])


    # Create an image to draw the lines on 创建一个图像来绘制线条
    # warp_zero = np.zeros_like(warped).astype(np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions
    color_warp = np.zeros((undist.shape[0], undist.shape[1]+500, 3), dtype='uint8')

    # Recast the x and y points into usable format for cv2.fillPoly()
    # 将x和y点重新转换为cv2.fillPoly（）的可用格式
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # 将逆透视变换后的车道线坐标点坐标转换为可用格式。
    pts_inv_left = np.array([np.transpose(np.vstack([left_point_inv_xx, left_point_inv_yy]))])
    pts_inv_right = np.array([np.flipud(np.transpose(np.vstack([right_point_inv_xx, right_point_inv_yy])))])
    pts_inv = np.hstack((pts_inv_left, pts_inv_right))

    # PMH 画车道线 参数：输入图像，点坐标，是否封闭，颜色，线宽，线类型。cv2.LINE_AA是反锯齿线，画曲线比较平滑
    cv2.polylines(color_warp, np.int_([pts]), False, (255, 0, 0), 10, cv2.LINE_AA)

    for ai in a:
        cv2.circle(color_warp, (np.int_(ai[0]), np.int_(ai[1])), 10, (0, 0, 255), -1)        # 绘制车道线两端圆点

    # cv2.imshow('polylines', color_warp)
    # Draw the lane onto the warped blank image将车道绘制到变形的空白图像上
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # cv2.imshow('color_warp', color_warp)

    # result = cv2.addWeighted(undist, 1, color_warp, 0.3, 0)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    # 使用反向透视矩阵（Minv）将空白转回原始图像空间
    newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image 将结果与原始图像组合在一起
    # cv2.imwrite('test.jpg',newwarp)
    newwarpNO = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
    newwarp[0:int(yy1[2]), int(xx1[0]):int(xx1[2])] = [0, 255, 0]

    if undist.shape.__len__() == 3:
        result = cv2.addWeighted(undist, 1, newwarpNO, 0.3, 0)
    else:
        out_img = (np.dstack((undist, undist, undist)) * 255).astype('uint8')
        result = cv2.addWeighted(out_img, 1, newwarpNO, 0.3, 0)

    # 在结果图上绘制车道线
    for i in range(4):
        cv2.circle(result, (int(xx1[i]), int(yy1[i])), 10, (0, 255, 255), -1)        # 绘制车道线两端圆点

    cv2.polylines(result, np.int_([pts_inv]), False, (0, 0, 255), 6, cv2.LINE_AA)
    # cv2.imshow('result', result)
    # Annotate lane curvature values and vehicle offset from center
    # 从中心注释车道曲率值和车辆偏移量
    # avg_curve = (left_curve + right_curve)/2
    # label_str = 'Radius of curvature: %.1f m' % avg_curve
    # result = cv2.putText(result, label_str, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)

    # label_str = 'Vehicle offset from lane center: %.1f m' % vehicle_offset
    # result = cv2.putText(result, label_str, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)

    # 显示过滤情况
    if is_left_filtered is True:
        label_str = 'Left lane is Filtered!'
        result = cv2.putText(result, label_str, (30, 90), 0, 2, (255, 0, 0), 6, cv2.LINE_AA)

    if is_right_filtered is True:
        label_str = 'Right lane is Filtered!'
        result = cv2.putText(result, label_str, (1030, 90), 0, 2, (255, 0, 0), 6, cv2.LINE_AA)

    return result,color_warp, newwarp, newwarpNO

