import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from skimage import img_as_ubyte
from skimage import data, color
from skimage.filter import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from scipy.ndimage import label
import scipy as sp

from declarations import DetectionMethod


class Detection:

	COIN_BOUND = 5
	HOUGH_MIN_RADIUS = 70
	HOUGH_MAX_RADIUS = 150
	DEFAULT_IMAGE_SIZE = 600
	DEFAULT_IMAGE_WIDTH = 1000
	DEFAULT_IMAGE_HEIGHT = 600
	MIN_CIRCLE_RATIO = 0.9
	MAX_CIRCLE_RATIO = 1.1
	INTERNAL_THRESHOLD = 40
	MIN_DIAMETER = 25
	MIN_AREA = 5
	MAX_AREA = 100000

	img = None

	def use_detection(self, img_filename, detection_method, img=None):
		if detection_method == DetectionMethod.WATERSHED:
			rects, ellipses = self.watershed(img_filename, img)
		elif detection_method == DetectionMethod.HOUGH_TRANSFORM:
			rects, ellipses = self.hough_transform(img_filename, img)
		elif detection_method == DetectionMethod.CANNY:
			rects, ellipses = self.canny(img_filename, img)
		elif detection_method == DetectionMethod.ALL:
			rects, ellipses = self.use_all(img_filename, img)
		return self.find_result(img_filename, rects, ellipses, img)


	def detect_coins(self, img_filename, detection_method, img=None, ret_coordinates=False):
		if not ret_coordinates:
			return self.use_detection(img_filename, detection_method, img)[0]
		else:
			return self.use_detection(img_filename, detection_method, img)

	def use_all(self, img_filename, img=None):
		res_rects, res_ellipses = [], []

		rects, ellipses = self.canny(img_filename, img)
		res_rects, res_ellipses = res_rects + rects, res_ellipses + ellipses

		rects, ellipses = self.watershed(img_filename, img)
		res_rects, res_ellipses = res_rects + rects, res_ellipses + ellipses

		rects, ellipses = self.hough_transform(img_filename, img)
		res_rects, res_ellipses = res_rects + rects, res_ellipses + ellipses

		return res_rects, res_ellipses

	def find_result(self, img_filename, rectangles, ellipses, img=None):
		if img is None:
			img = cv2.imread(img_filename)
			self.img = img

		rectangles, ellipses = self.remove_overlaping_rects(rectangles, ellipses)
		for ellipse in ellipses:
			cv2.ellipse(img, ellipse, (0,255,0), 2)
		return img, rectangles

	def crop_detected_coins(self, img_filename, detection_method):
		img, rectangles = self.use_detection(img_filename, detection_method)
		img_clean = cv2.imread(img_filename)
		img_h, img_w, d = img.shape
		detected_coins = []
		for rect in rectangles:
			x,y,w,h = rect
			# enlarge rectangle a little
			w = w+self.COIN_BOUND if x+w+self.COIN_BOUND <= img_w else img_w-x
			h = h+self.COIN_BOUND if y+h+self.COIN_BOUND <= img_h else img_h-y
			x, w = (x-self.COIN_BOUND, w+self.COIN_BOUND) if x-self.COIN_BOUND >= 0 else (0, w+x)
			y, h = (y-self.COIN_BOUND, h+self.COIN_BOUND) if y-self.COIN_BOUND >= 0 else (0, h+y)
			detected_coins.append(img_clean[y : y+h, x : x+w])
		return img, detected_coins

	def canny(self, img_filename, img=None):
		if img is None:
			img = cv2.imread(img_filename)
			self.img = img

		print "img.shape = {}".format(img.shape)
		h, w, d = img.shape
		if h > self.DEFAULT_IMAGE_HEIGHT or w > self.DEFAULT_IMAGE_WIDTH:
			if h > self.DEFAULT_IMAGE_HEIGHT:
				new_h = self.DEFAULT_IMAGE_HEIGHT
				new_w = w * new_h / h
				w, h = new_w, new_h
			if w > self.DEFAULT_IMAGE_WIDTH:
				new_w = self.DEFAULT_IMAGE_WIDTH
				new_h = h * new_w / w
				w, h = new_w, new_h
			print "new_w = {}, new_h = {}".format(w, h)
			img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
		print "new img.shape = {}".format(img.shape)

		# convert to grayscale and detect edges
		gray = color.rgb2gray(img)
		edges = canny(gray, sigma=2.2, low_threshold=0.085, high_threshold=0.3)

		gray_blur = img_as_ubyte(edges)
		closing = gray_blur


		# Contour detection and filtering
		cont_img = closing.copy()
		contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
		    cv2.CHAIN_APPROX_SIMPLE)

		adjust_contours = []
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area < self.MIN_AREA or area > self.MAX_AREA:
				continue
			if len(cnt) < 5:
			    continue
			adjust_contours.append(cnt)
		return self.convert_contours(img, adjust_contours)

	def watershed(self, img_filename, img=None):
		if img is None:
			img = cv2.imread(img_filename)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		c = 3
		block_size = max(gray.shape[:2]) / c
		block_size = block_size + 1 if not block_size%2 else block_size
		ad_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, \
			block_size, 2)

		kernel = np.ones((2, 2), np.uint8)
		opening = cv2.erode(ad_thresh, kernel, iterations = 1)
		kernel = np.ones((10, 10), np.uint8)
		closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,
		    kernel, iterations=7)

		# Finding sure background area
		sure_bg = cv2.dilate(closing,kernel,iterations=3)

		# Finding sure foreground area
		kernel = np.ones((5, 5), np.uint8)
		prepare = cv2.morphologyEx(ad_thresh, cv2.MORPH_CLOSE,
		    kernel, iterations=5)
		dist_transform = cv2.distanceTransform(prepare,cv2.cv.CV_DIST_L2,5)

		ret, sure_fg = cv2.threshold(dist_transform,0.48*dist_transform.max(),255,0)

		# Finding unknown region
		sure_fg = np.uint8(sure_fg)
		unknown = cv2.subtract(sure_bg,sure_fg)

		# Lable regions
		lbl, ncc = label(sure_fg)
		lbl = lbl+1
		lbl[unknown==255] = 0
		lbl = lbl.astype(np.int32)

		# Apply watershed
		# Add blur
		b_img = cv2.medianBlur(img,1)
		cv2.watershed(b_img,lbl)
		lbl[lbl == -1] = 0
		lbl = lbl.astype(np.uint8)
		result = 255 - lbl
		result[result != 255] = 0

		#remove borders
		for i in range(result.shape[1]):
				result[0][i] = 0
				result[result.shape[0]-1][i] = 0
		for i in range(result.shape[0]):
				result[i][0] = 0
				result[i][result.shape[1]-1] = 0

		kernel = np.ones((3, 3), np.uint8)
		result = cv2.dilate(result, kernel, iterations=1)

		# Contour detection and filtering
		cont_img = result
		result[result == 255] = 1
		contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
		    cv2.CHAIN_APPROX_SIMPLE)

		return self.convert_contours(img, contours)

	def hough_transform(self, img_filename, img=None):
		if img is None:
			img = cv2.imread(img_filename)

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)

		circles = cv2.HoughCircles(image=gray_blur, method=cv2.cv.CV_HOUGH_GRADIENT, dp=1,
			minDist=2*self.HOUGH_MIN_RADIUS, param1=55, param2=60,
			minRadius=self.HOUGH_MIN_RADIUS, maxRadius=self.HOUGH_MAX_RADIUS)

		rectangles = []
		ellipses = []
		if circles is not None:
			circles = np.uint16(np.around(circles))
			for i in circles[0,:]:
			    rectangles.append((i[0]-i[2], i[1]-i[2], 2*i[2], 2*i[2]))
			    ellipses.append(((i[0], i[1]), (2*i[2], 2*i[2]), 0))
		return rectangles, ellipses

	def remove_overlaping_rects(self, rects, ellipses):
		new_rects = []
		new_ellipses = []
		for i in range(len(ellipses)):
			center, axes, theta = ellipses[i]
			x1, y1 = center[0] - axes[0]/2, center[1] - axes[1]/2
			x2, y2 = center[0] + axes[0]/2, center[1] + axes[1]/2
			if x2 - x1 > self.img.shape[0] or y2 - y1 > self.img.shape[1]:
				continue
			new_ellipses.append(ellipses[i])
			new_rects.append(rects[i])
		rects, ellipses = new_rects, new_ellipses

		new_rects, new_ellipses = [], []
		for i in range(len(ellipses)):
			center, axes, theta = ellipses[i]
			x1, y1 = center[0] - axes[0]/2, center[1] - axes[1]/2
			x2, y2 = center[0] + axes[0]/2, center[1] + axes[1]/2
			outer_rect = True
			for j in range(len(ellipses)):
				if i != j and ellipses[i] != ellipses[j]:
					center, axes, theta = ellipses[j]
					x1j, y1j = center[0] - axes[0]/2, center[1] - axes[1]/2
					x2j, y2j = center[0] + axes[0]/2, center[1] + axes[1]/2
					if (x1j - x1 <= self.INTERNAL_THRESHOLD and y1j - y1 <= self.INTERNAL_THRESHOLD and
							x2 - x2j <= self.INTERNAL_THRESHOLD and y2 - y2j <= self.INTERNAL_THRESHOLD):
						if x1 > x1j and y1 > y1j and x2j > x2 and y2j > y2:
								ellipses[i] = ellipses[j]
								outer_rect = False
								break
						if x1 < x1j and y1 < y1j and x2j < x2 and y2j < y2:
								continue
						if (x1 - x1j <= self.INTERNAL_THRESHOLD and y1 - y1j <= self.INTERNAL_THRESHOLD and
								x2j - x2 <= self.INTERNAL_THRESHOLD and y2j - y2 <= self.INTERNAL_THRESHOLD):
							w, h = x2 - x1, y2 - y1
							wj, hj = x2j - x1j, y2j - y1j
							if abs(wj/hj - 1) < abs(w/h - 1):
								ellipses[i] = ellipses[j]
								outer_rect = False
								break
							continue
						outer_rect = False
						break
			if outer_rect:
				new_rects.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
				new_ellipses.append(ellipses[i])

		return new_rects, new_ellipses

	def convert_contours(self, img, contours):
		rectangles = []
		ellipses = []
		for cnt in contours:
			area = cv2.contourArea(cnt)
			ellipse = cv2.fitEllipse(cnt)
			center, size, theta = ellipse
			w, h = size
			if w/h > self.MAX_CIRCLE_RATIO or w/h < self.MIN_CIRCLE_RATIO:
				continue
			if w < self.MIN_DIAMETER or h < self.MIN_DIAMETER:
				continue
			rectangles.append(cv2.boundingRect(cnt))
			ellipses.append(ellipse)

		return rectangles, ellipses

	def draw_matches(self, img1, kp1, img2, kp2, matches):
		h1, w1 = img1.shape[:2]
		h2, w2 = img2.shape[:2]
		view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
		view[:h1, :w1, 0] = img1
		view[:h2, w1:, 0] = img2
		view[:, :, 1] = view[:, :, 0]
		view[:, :, 2] = view[:, :, 0]

		for m in matches:
		    color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
		    cv2.line(view, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])) , (int(kp2[m.trainIdx].pt[0] + w1),
		    	int(kp2[m.trainIdx].pt[1])), color, thickness = 2)

		return view
