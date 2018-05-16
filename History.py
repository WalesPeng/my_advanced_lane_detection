import numpy as np


class History():
    def __init__(self, n: object) -> object:
        """
		n is the window size of the moving average
		"""
        self.n = n
        self.detected = False

        # Polynomial coefficients: x = A*y^2 + B*y + C
        # Each of A, B, C is a "list-queue" with max length n
        self.data = []
        self.smoothed = []
        self.data = []
        self.coeffs = []
        self.smoothed_coeffs = []
        self.midfilter_data = []
        self.latest = 0
        self.latestSmoothed = 0
        self.latestA = 0
        self.latestB = 0
        self.latestC = 0

    def get_latest(self):
        """
        获得最近的一帧历史数据
        """
        self.latest = self.data[len(self.data) - 1]
        return self.latest

    def get_smoothed(self):
        """
        获得最近的一帧历史数据
        """
        self.latestSmoothed = self.smoothed_coeffs[len(self.smoothed_coeffs) - 1]
        return self.latestSmoothed

    def update_history(self, coeffs):
        """
        # predict current lane parameter
        # 更新历史数据
        """
        # Coefficient queue full?系数队列已满？
        q_full = len(self.data) >= self.n

        # Append line fit coefficients追加线条拟合系数
        self.data.append(coeffs)

        # Pop from index 0 if full
        if q_full:
            _ = self.data.pop(0)

        return self

    def add_smoothing(self, alpha):
        """
        # predict current lane parameter
        # 采用 exponential_smoothing 算法，根据历史信息预测当前车道线曲率和x轴截距
        """
        self.smoothed = np.zeros(len(self.data))
        self.smoothed[0] = self.data[0]
        for i in range(1, len(self.smoothed)):
            self.smoothed[i] = alpha*self.data[i]+(1-alpha)*self.smoothed[i-1]

        return self


    def get_mid(self):
        """
        # predict current lane parameter
        # 更新历史数据
        """
        self.midfilter_data = self.data.copy()

        # Coefficient queue full?系数队列已满？
        q_full = len(self.data) < 5

        # same padding
        if q_full:
            self.midfilter_data.reverse()
            while len(self.midfilter_data) < 5:
                self.midfilter_data.append(self.data[0])

            self.midfilter_data.reverse()

        for i in range(3):
            min = i
            for k in range(i+1, 5):
                if self.midfilter_data[k] < self.midfilter_data[min]:

                    min = k

            temp = self.midfilter_data[i]
            self.midfilter_data[i] = self.midfilter_data[min]
            self.midfilter_data[min] = temp

        mid_value = self.midfilter_data[2]

        return mid_value

    def smooth_coeffs(self, alpha):
        """
        # predict current lane parameter
        # 采用 exponential_smoothing 算法，根据历史信息预测当前车道线曲率和x轴截距
        """
        self.smoothed_coeffs = np.zeros((len(self.data),3))
        self.smoothed_coeffs[0] = self.data[0]
        for i in range(1, len(self.smoothed_coeffs)):
            self.smoothed_coeffs[i][0] = alpha * self.data[i][0] + (1 - alpha) * self.smoothed_coeffs[i - 1][0]
            self.smoothed_coeffs[i][1] = alpha * self.data[i][1] + (1 - alpha) * self.smoothed_coeffs[i - 1][1]
            self.smoothed_coeffs[i][2] = alpha * self.data[i][2] + (1 - alpha) * self.smoothed_coeffs[i - 1][2]

        return self

    # def get_history_coeffs(self):
    #     """
    #     获得最近的一帧历史数据的车道线系数信息
    #     """
    #
    #     self.latestC = self.coeffs[len(self.coeffs) - 1][2]
    #     self.latestB = self.coeffs[len(self.coeffs) - 1][1]
    #     self.latestA = self.coeffs[len(self.coeffs) - 1][0]
    #     return (self.latestA, self.latestB, self.latestC)