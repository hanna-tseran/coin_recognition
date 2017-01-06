import cv2
import cv
import numpy as np

import util
from declarations import DetectionMethod
from detection import Detection

class Rect:
	center = []
	bounds = []
	def __init__(self, center, bounds):
		self.center = center
		self.bounds = bounds


class DetectionQuality:

	DATABASE_PATH = "detection quality/"
	GROUND_TRUTH = "ground truth/"
	TEST = "test/"
	detection = Detection()
	gt_detection_method = DetectionMethod.HOUGH_TRANSFORM

	def measure_quality(self, detection_method):
		print "path = {}".format(self.DATABASE_PATH + self.GROUND_TRUTH)
		filenames_gt = util.get_imlist(self.DATABASE_PATH + self.GROUND_TRUTH)
		filenames_test = util.get_imlist(self.DATABASE_PATH + self.TEST)
		print filenames_gt
		print "filenames.size = {}".format(len(filenames_gt))

		white = np.asarray([255,255,255])
		# self.detection.hough_min_radius = 10
		for img_i in xrange(len(filenames_gt)):
			# leave only black pixels
			img_gt = cv2.imread(filenames_gt[img_i])
			for i in range(img_gt.shape[0]):
				for j in range(img_gt.shape[1]):
					if (img_gt[i][j] > 15).any():
						img_gt[i][j] = white
			img_gt, bounds_gt = self.detection.detect_coins("", self.gt_detection_method, img_gt, True)
			print "bounds_gt:"
			print bounds_gt
			cv2.imshow("ground truth img{}".format(img_i), img_gt)
			rects_gt = []
			for rect in bounds_gt:
				rects_gt.append(Rect([(rect[2]-rect[0]) / 2, (rect[3]-rect[1]) / 2],rect))
			print "rects_gt:"
			for rect in rects_gt:
				print "center = {}, bounds = {}".format(rect.center, rect.bounds)

			img_test = cv2.imread(filenames_test[img_i])
			img_test, rect_test = self.detection.detect_coins("", detection_method, img_test, True)
			print "rect_test:"
			print rect_test
			# cv2.imshow("detected coins img{}".format(img_i), img_test)

			with open("img_0.txt", 'r') as file:
				for line in file:
					print line

