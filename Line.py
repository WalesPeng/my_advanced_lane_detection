import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# Define a class to receive the characteristics of each line detection
class Line():
	def __init__(self, n):
		"""
		n is the window size of the moving average
		"""
		self.n = n
		self.detected = False

		# Polynomial coefficients: x = A*y^2 + B*y + C
		# Each of A, B, C is a "list-queue" with max length n
		self.A = []
		self.B = []
		self.C = []
		# Average of above
		self.A_avg = 0.
		self.B_avg = 0.
		self.C_avg = 0.

	def get_fit(self):
		return (self.A_avg, self.B_avg, self.C_avg)

	def add_fit(self, fit_coeffs):
		"""
		Gets most recent line fit coefficients and updates internal smoothed coefficients
		fit_coeffs is a 3-element list of 2nd-order polynomial coefficients
		获取最新的线拟合系数和更新内部平滑系数fit_coeffs是二元多项式系数的3元素列表
		"""
		# Coefficient queue full?系数队列已满？
		q_full = len(self.A) >= self.n

		# Append line fit coefficients追加线条拟合系数
		self.A.append(fit_coeffs[0])
		self.B.append(fit_coeffs[1])
		self.C.append(fit_coeffs[2])

		# Pop from index 0 if full
		if q_full:
			_ = self.A.pop(0)
			_ = self.B.pop(0)
			_ = self.C.pop(0)


		# # Simple average of line coefficients
		# self.A_avg = self.A[0] * 0.1 + self.A[1] * 0.2 + self.A[2] * 0.3 + self.A[3] * 0.4
		# self.B_avg = self.B[0] * 0.1 + self.B[1] * 0.2 + self.B[2] * 0.3 + self.B[3] * 0.4
		# self.C_avg = self.C[0] * 0.1 + self.C[1] * 0.2 + self.C[2] * 0.3 + self.C[3] * 0.4
		self.A_avg = fit_coeffs[0]
		self.B_avg = fit_coeffs[1]
		self.C_avg = fit_coeffs[2]

		return (self.A_avg, self.B_avg, self.C_avg)


