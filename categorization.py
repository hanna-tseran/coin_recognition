import os
import re
import sys
from random import shuffle
import time

import cv2

import numpy as np

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skimage.filter import canny
from skimage import img_as_ubyte

from nolearn.lasagne import NeuralNet
from lasagne import layers
from lasagne import nonlinearities
from lasagne.updates import nesterov_momentum

from declarations import DetectionMethod, ClassifierType
from detection import Detection
import util
import pickle


class Categorization:

	DATABASE_PATH = "database/"
	TRAIN_DATA_PATH = "database/train/"
	TEST_DATA_PATH = "database/test/"
	TRAIN_FOLDERS = ["1ruble", "2rubles", "5rubles", "10rubles", "heads_ruble",
						"1kopeck", "5kopecks", "10kopecks", "50kopecks", "heads_kopeck"]
	CLASS_NAMES = ["1 ruble", "2 rubles", "5 rubles", "10 rubles", "heads ruble",
						"1 kopeck", "5 kopecks", "10 kopecks", "50 kopecks", "heads kopeck"]
	TEST_FILES = ["test1r.jpg", "test2r.jpg", "test5r.jpg", "test10r.jpg", "testhr.jpg",
						"test1k.jpg", "test5k.jpg", "test10k.jpg", "test50k.jpg", "testhk.jpg"]
	EXPECTED_CLASSES = [0, 1, 2, 3, 4, 5, 6 ,7, 8, 9]
	DEFAULT_IMAGE_SIZE = (235, 235)
	CLASS_TRAIN_IMAGES_NUM = 10
	FIND_BEST = False#True

	knn_neural_net = None
	neural_net = None
	knn_svm_linear = None
	svm_linear =  None
	knn_random_forest = None
	random_forest =  None

	K_RBF = 700
	NEIGHBORS_NUM_RBF = 50
	K_LINEAR = 400
	NEIGHBORS_NUM_LINEAR = 50
	GAMMA = 0.0
	MAX_DEPTH = 50
	N_ESTIMATORS = 300
	MAX_FEATURES = "log2"
	K_RANDOM_FOREST = 600
	NEIGHBORS_NUM_RANDOM_FOREST = 50

	NN_PIXELS = 96#235

	sift = cv2.SIFT()

	model_learned = False
	knn = None
	svm = None

	K = 1400
	NEIGHBORS_NUM = 110

	K_NN = 1400
	NEIGHBORS_NUM_NN = 110

	CLASSIFIER = ClassifierType.NEURAL_NETWORK

	IMAGES_FOR_CENTER = 2
	NEED_DUMPING = False#True


	def categorize_coins(self, img_filename, detection_method, classifier_type=None):
		if self.FIND_BEST:
			find_best_svm(pickle.load(open("training_descriptors.p", "rb")))

		self.NEURAL_NETWORK_LOADED = False
		self.SVM_LINEAR_LOADED = False
		self.RANDOM_FOREST_LOADED = False

		if classifier_type == None:
			classifier_type = self.CLASSIFIER
		print "ClassifierType == {}".format(classifier_type)

		detection = Detection()
		detected_coins_img, coins = detection.crop_detected_coins(img_filename, detection_method)

		need_loading = True
		if classifier_type == ClassifierType.NEURAL_NETWORK:
			self.K = self.K_NN
			self.NEIGHBORS_NUM = self.NEIGHBORS_NUM_NN
			if self.NEURAL_NETWORK_LOADED:
				need_loading = False
			else:
				self.NEURAL_NETWORK_LOADED = True

		elif classifier_type == ClassifierType.SVM_LINEAR:
			self.K = self.K_LINEAR
			self.NEIGHBORS_NUM = self.NEIGHBORS_NUM_LINEAR
			if self.SVM_LINEAR_LOADED:
				need_loading = False
			else:
				self.SVM_LINEAR_LOADED = True

		elif classifier_type == ClassifierType.RANDOM_FOREST:
			self.K = self.K_RANDOM_FOREST
			self.NEIGHBORS_NUM = self.NEIGHBORS_NUM_RANDOM_FOREST
			if self.RANDOM_FOREST_LOADED:
				need_loading = False
			else:
				self.RANDOM_FOREST_LOADED = True

		if self.NEED_DUMPING:
			# training_descriptors = self.get_sift_training()
			# pickle.dump(training_descriptors, open("training_descriptors.p", "wb"))
			training_descriptors = pickle.load(open("training_descriptors.p", "rb"))

			# main_descriptors = self.get_main_descriptors(training_descriptors)
			# pickle.dump(main_descriptors, open("main_descriptors.p", "wb"))
			main_descriptors = pickle.load(open("main_descriptors.p", "rb"))

			# Group similar descriptors into clusters
			cluster_centers = self.get_similar_descriptors(self.K, main_descriptors)

			# Compute training data for SVM classifier
			all_histograms, final_labels, self.knn = self.compute_training_data(
				self.K, cluster_centers, training_descriptors)

			if classifier_type == ClassifierType.NEURAL_NETWORK:
				pickle.dump(self.knn, open("knn_neural_net.p", "wb"))
			elif classifier_type == ClassifierType.SVM_LINEAR:
				pickle.dump(self.knn, open("knn_svm_linear_svm_linear.p", "wb"))
			elif classifier_type == ClassifierType.RANDOM_FOREST:
				pickle.dump(self.knn, open("knn_random_forest.p", "wb"))

			self.classifier = self.train_classifier(all_histograms, final_labels, classifier_type)

			if classifier_type == ClassifierType.NEURAL_NETWORK:
				pickle.dump(self.classifier, open("neural_net.p", "wb"))
			elif classifier_type == ClassifierType.SVM_LINEAR:
				pickle.dump(self.classifier, open("svm_linear.p", "wb"))
			elif classifier_type == ClassifierType.RANDOM_FOREST:
				pickle.dump(self.classifier, open("random_forest.p", "wb"))
			return

		if need_loading:
			if classifier_type == ClassifierType.NEURAL_NETWORK:
				self.knn_neural_net = pickle.load(open("knn_neural_net.p", "rb"))
				self.neural_net = pickle.load(open("neural_net.p", "rb"))
			elif classifier_type == ClassifierType.SVM_LINEAR:
				self.knn_svm_linear = pickle.load(open("knn_svm_linear.p", "rb"))
				self.svm_linear = pickle.load(open("svm_linear.p", "rb"))
			elif classifier_type == ClassifierType.RANDOM_FOREST:
				self.knn_random_forest = pickle.load(open("knn_random_forest.p", "rb"))
				self.random_forest = pickle.load(open("random_forest.p", "rb"))

		if classifier_type == ClassifierType.NEURAL_NETWORK:
			self.classifier = self.neural_net
			self.knn = self.knn_neural_net
		elif classifier_type == ClassifierType.SVM_LINEAR:
			self.classifier = self.svm_linear
			self.knn = self.knn_svm_linear
		elif classifier_type == ClassifierType.RANDOM_FOREST:
			self.classifier = self.random_forest
			self.knn = self.knn_random_forest

		descriptors = self.get_sift_descriptors(coins)
		predicted_labels = self.classify(coins, self.K, self.knn, descriptors, self.classifier)

		# Show images with predicted labels
		self.show_predicted(coins, predicted_labels, self.CLASS_NAMES)

	# compute SIFT descriptors, that will be used to compute cluster centers
	def get_main_descriptors(self, training_descriptors):
		main_descriptors = []
		for cls in training_descriptors:
			img_i = 0
			for img in cls:
				for descr in img:
					main_descriptors.append(descr)
				img_i += 1
				if img_i >= self.IMAGES_FOR_CENTER:
					break
		main_descriptors = np.asarray(main_descriptors)
		return main_descriptors

	def compute_SIFT_descriptors(self):
		filenames = util.get_imlist(self.DATABASE_PATH)

		descriptor_mat=[]

		for img_i in xrange(len(filenames)):
			img = cv2.imread(filenames[img_i])
			img = cv2.resize(img, self.DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			gray = cv2.equalizeHist(gray)

			# Detect SIFT keypoints and compute descriptors
			kp, des = self.sift.detectAndCompute(gray, None)
			descriptor_mat.append(des)
		return descriptor_mat

	def get_similar_descriptors(self, k, descriptor_mat):
		descriptor_mat = np.double(np.vstack(descriptor_mat))
		print "true descriptor_mat.shape: {}".format(descriptor_mat.shape)

		# Group descriptors into k clusters.
		kmeans = KMeans(k)
		kmeans.fit(descriptor_mat)

		# Get cluster centers
		cluster_centers = kmeans.cluster_centers_
		return cluster_centers

	def show_training_images(self):
		training_sample = []
		for folder in self.TRAIN_FOLDERS:
			# Get all ttraining images from particular class
			filenames = util.get_imlist("{0}/{1}".format(self.TRAIN_DATA_PATH, folder))
			for i in xrange(len(filenames)):
				temp = cv2.imread(filenames[i])
				training_sample.append(temp)

		for class_num in xrange(len(self.TRAIN_FOLDERS)):
			plt.rcParams["figure.figsize"] = 10,6
			fig = plt.figure()
			plt.title("10 training images for class {}".format(class_num))
			for image_no in xrange(self.CLASS_TRAIN_IMAGES_NUM):
				fig.add_subplot(self.CLASS_TRAIN_IMAGES_NUM/5, 5, image_no+1)
				self.showfig(training_sample[class_num*self.CLASS_TRAIN_IMAGES_NUM+image_no], None)
			plt.show()

	def get_sift_training(self, img_for_class=None):
		folder_number = 0
		des_training = []
		for folder in self.TRAIN_FOLDERS:
			# Get all training images from a particular class
			filenames = util.get_imlist("{0}/{1}".format(self.TRAIN_DATA_PATH, folder))
			filenames = np.array(filenames)
			if img_for_class:
				max_img_num = img_for_class
			else:
				max_img_num = len(filenames)

			des_per_folder = []
			img_num = 0
			for image_name in filenames:
				img = cv2.imread(image_name)

				# Preprocessing
				gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				gray = cv2.resize(gray, self.DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
				gray = cv2.equalizeHist(gray)

				# Get all SIFT descriptors for an image
				_, des = self.sift.detectAndCompute(gray, None)
				des_per_folder.append(des)

				img_num += 1
				if img_num >= max_img_num:
					break

			des_training.append(des_per_folder)
			folder_number += 1
		return des_training

	def compute_training_data(self, k, cluster_centers, descriptors):
		all_histograms = []
		# labels for all of the test images
		final_labels = []
		# to hold the cluster number a descriptor belong to
		cluster_labels = []

		# initialize KNN
		knn = KNeighborsClassifier(n_neighbors=k)

		# Target descriptors are the cluster_centers that we got earlier.
		# All the descriptors of an image are matched against these for
		# calculating the histogram.
		knn.fit(cluster_centers, range(k))

		folder_number = 0
		for folder in self.TRAIN_FOLDERS:
			# get all the training images from a particular class
			filenames = util.get_imlist("{0}/{1}".format(self.TRAIN_DATA_PATH, folder))
			img_num = 0
			for image_name in xrange(len(filenames)):
				des = descriptors[folder_number][image_name]

				# find all the labels of cluster_centers that are nearest
				# to the descriptors present in the current image.
				cluster_labels = []
				for i in xrange(len(des)):
					cluster_labels.append(knn.kneighbors(des[i],
						n_neighbors=self.NEIGHBORS_NUM, return_distance=False)[0][0])

				histogram_per_image=[]
				for i in xrange(k):
				    histogram_per_image.append(cluster_labels.count(i))

				all_histograms.append(np.array(histogram_per_image))
				final_labels.append(folder_number)

				img_num += 1
				if img_num >= len(descriptors[folder_number]):
					break

			folder_number += 1

		all_histograms = np.array(all_histograms)
		final_labels = np.array(final_labels)
		return all_histograms, final_labels, knn

	def train_classifier(self, all_histograms, final_labels, classifier_type):
		if classifier_type == ClassifierType.SVM_RBF:
			return self.train_svm_rbf(all_histograms, final_labels)
		elif classifier_type == ClassifierType.SVM_LINEAR:
			return self.train_svm_linear(all_histograms, final_labels)
		elif classifier_type == ClassifierType.RANDOM_FOREST:
			return self.train_random_forest(all_histograms, final_labels)
		elif classifier_type == ClassifierType.NEURAL_NETWORK:
			return self.train_neural_net(all_histograms, final_labels)

	def train_svm_rbf(self, all_histograms, final_labels):
		svm = SVC(kernel = "rbf", gamma = self.GAMMA)
		svm.fit(all_histograms, final_labels)
		return svm

	def train_svm_linear(self, all_histograms, final_labels):
		svm = SVC(kernel = "linear", gamma = self.GAMMA)
		svm.fit(all_histograms, final_labels)
		return svm

	def train_random_forest(self, all_histograms, final_labels):
		rf = RandomForestClassifier(max_depth=self.MAX_DEPTH, n_estimators=self.N_ESTIMATORS,
			max_features=self.MAX_FEATURES)
		rf.fit(all_histograms, final_labels)
		return rf

	def train_svm(self, all_histograms, final_labels):
		svm = SVC(kernel = self.KERNEL, gamma = self.GAMMA)
		svm.fit(all_histograms, final_labels)
		return svm

	def folder_to_class(self, folder):
		if folder == '1ruble':
			return 0
		elif folder == '2rubles':
			return 1
		elif folder == '5rubles':
			return 2
		elif folder == '10rubles':
			return 3
		elif folder == 'heads_ruble':
			return 4
		elif folder == '1kopeck':
			return 5
		elif folder == '5kopecks':
			return 6
		elif folder == '10kopecks':
			return 7
		elif folder == '50kopecks':
			return 8
		elif folder == 'heads_kopeck':
			return 9

	def train_neural_net(self, all_histograms, final_labels):

		net = NeuralNet(
			layers=[
				('input', layers.InputLayer),
				('hidden1', layers.DenseLayer),
				('dropout1', layers.DropoutLayer),
				('hidden2', layers.DenseLayer),
				('dropout2', layers.DropoutLayer),
				('hidden3', layers.DenseLayer),
				('dropout3', layers.DropoutLayer),
				('hidden4', layers.DenseLayer),
				('output', layers.DenseLayer),
				],
			input_shape=(None, all_histograms[0].shape[0]),
			hidden1_num_units=500,
			dropout1_p=0.1,
			hidden2_num_units=1000,
			dropout2_p=0.2,
			hidden3_num_units=500,
			dropout3_p=0.3,
			hidden4_num_units=300,
			output_num_units=10, output_nonlinearity=nonlinearities.softmax,
			update=nesterov_momentum,
			update_learning_rate=0.001,
			update_momentum=0.9,

			eval_size=0.2,
			verbose=1,
			max_epochs=500
		)
		all_histograms_sh = []
		final_labels_sh = []
		index_shuffled = range(len(final_labels))
		shuffle(index_shuffled)
		for i in index_shuffled:
		    all_histograms_sh.append(all_histograms[i])
		    final_labels_sh.append(final_labels[i])
		all_histograms = all_histograms_sh
		final_labels = final_labels_sh

		all_histograms = np.array(all_histograms).astype(np.int32)
		final_labels = np.array(final_labels).astype(np.int32)
		net.fit(all_histograms, final_labels)
		accuracy = sum([i['valid_accuracy'] for i in net.train_history_][-50:]) / 50
		print "accuracy for NN: {}".format(accuracy)
		return net

	def classify_with_neural_net(self, net, coins):
		converted_coins = []
		for coin in coins:
			coin = cv2.resize(coin, (self.NN_PIXELS, self.NN_PIXELS), interpolation=cv2.INTER_AREA)
			coin = cv2.cvtColor(coin,cv2.COLOR_BGR2GRAY)
			coin = np.array(coin)
			converted_coins.append(np.reshape(coin, (self.NN_PIXELS * self.NN_PIXELS)))
		converted_coins = np.array(converted_coins)

		predicted_labels = net.predict(converted_coins)
		print "predicted_labels: {}".format(predicted_labels)
		return predicted_labels

	def show_testing_images(self, testing_images):
		plt.rcParams['figure.figsize'] = 20,8
		fig = plt.figure()
		plt.title('Test Images')
		for image_no in xrange(len(testing_images)):
			fig.add_subplot(3,8, image_no+1)
			self.showfig(testing_images[image_no], None)
		plt.show()

	def get_sift_descriptors(self, images):
		descriptors = []
		for img in images:
			result = []

			# perform preprocessing routines
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			gray = cv2.resize(gray, self.DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
			gray = cv2.equalizeHist(gray)

			# get SIFT descriptors for an image
			_, des = self.sift.detectAndCompute(gray, None)
			descriptors.append(des)
		return descriptors

	def get_sift_descriptors(self, images):
		descriptors = []
		for img in images:
			result = []

			# Preprocessing
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			gray = cv2.resize(gray, self.DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
			gray = cv2.equalizeHist(gray)

			# Get all SIFT descriptors for an image
			_, des = self.sift.detectAndCompute(gray, None)
			descriptors.append(des)
		return descriptors

	def classify(self, testing_images, k, knn, des_testing, classifier):
		all_histograms = []
		image_no = 0
		for img in testing_images:
			result = []
			des = des_testing[image_no]
			cluster_labels = []
			for i in xrange(len(des)):
				cluster_labels.append(knn.kneighbors(des[i],
					n_neighbors=self.NEIGHBORS_NUM, return_distance=False)[0][0])

			histogram = []
			for i in xrange(k):
				histogram.append(cluster_labels.count(i))

			all_histograms.append(np.array(histogram))
			image_no += 1

		all_histograms = np.double(np.array(all_histograms))
		return classifier.predict(all_histograms)


	def classify_with_svm(self, testing_images, k, knn, svm, des_testing):
		all_histograms = []
		image_no = 0
		for img in testing_images:
			result = []
			des = des_testing[image_no]

			cluster_labels = []
			for i in xrange(len(des)):
				cluster_labels.append(knn.kneighbors(des[i],
					n_neighbors=self.NEIGHBORS_NUM, return_distance=False)[0][0])

			histogram = []
			for i in xrange(k):
				histogram.append(cluster_labels.count(i))

			all_histograms.append(np.array(histogram))
			image_no += 1

		all_histograms = np.double(np.array(all_histograms))
		return svm.predict(all_histograms)

	def classify_with_random_forest(self, testing_images, k, knn, rf, des_testing):
		all_histograms = []
		image_no = 0
		for img in testing_images:
			result = []
			des = des_testing[image_no]
			cluster_labels = []
			for i in xrange(len(des)):
				cluster_labels.append(knn.kneighbors(des[i],
					n_neighbors=self.NEIGHBORS_NUM, return_distance=False)[0][0])
			histogram = []
			for i in xrange(k):
				histogram.append(cluster_labels.count(i))

			all_histograms.append(np.array(histogram))
			image_no += 1

		all_histograms = np.double(np.array(all_histograms))
		return rf.predict(all_histograms)

	def create_conf_matrix(self, expected, predicted, n_classes):
		m = [[0] * n_classes for i in range(n_classes)]
		for pred, exp in zip(predicted, expected):
			m[exp][int(pred)] += 1
		return np.array(m)

	def find_best_svm(self, training_descriptors):
		images_for_center = 2
		training_descriptors_centers = []
		for folder in training_descriptors:
			img_i = 0
			for img in folder:
				for descr in img:
					training_descriptors_centers.append(descr)
				img_i += 1
				if img_i >= images_for_center:
					break
		training_descriptors_centers = np.asarray(training_descriptors_centers)

		for k in [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200]:#[200, 400, 600, 1000, 1400, 2000]:
			self.K = k
			start = time.time()
			cluster_centers = self.get_similar_descriptors(k, training_descriptors_centers)
			end = time.time()
			print "for K: {} elapsed: {}".format(k, end-start)
		return

		max_accuracy = 0
		best_k = 100
		best_neighbors = 1
		best_kernel = 'linear'
		best_gamma = 0.0
		best_classifier = ClassifierType.SVM_LINEAR
		times_to_repeat = 5
		start_k = 100

		kernel = 'svm'
		gamma = 0.0

		# for clusters_num in [100, 200, 300, 400, 500]:
		# 	for test_k in [1, 3, 5, 10, 15]:

		# testing_images = []
		# for filename in test_filenames:
		# 	testing_images.append(cv2.imread("{0}{1}".format(self.TEST_DATA_PATH, filename)))
		# testing_descriptors = self.get_sift_testing(testing_images)

		for k in [200, 400, 600, 1000, 1400, 2000]:#[200, 400, 600, 1000, 1400, 2000]:
			self.K = k
			cluster_centers = self.get_similar_descriptors(k, training_descriptors_centers)
			for neighbors_num in [50, 70, 110, 150]:#[50, 70, 110, 150]:#[10, 30, 50, 70, 90, 110, 130, 150]:
				self.NEIGHBORS_NUM = neighbors_num
				all_histograms, final_labels, knn = self.compute_training_data(k,
									cluster_centers, training_descriptors)

				for classifier in [ClassifierType.NEURAL_NETWORK]:#ClassifierType.NEURAL_NETWORK]:#, ClassifierType.RANDOM_FOREST, ClassifierType.SVM_LINEAR]:
				#ClassifierType.RANDOM_FOREST, ClassifierType.NEURAL_NETWORK
					self.CLASSIFIER = classifier

					if classifier == ClassifierType.NEURAL_NETWORK:
						# for dense0_num_units in [k/3, k/2, k]:
						# 	for dropout_p in [0.3, 0.5, 0.7]:
						# 		for dense1_num_units in [k/4, k/3, k/2]:

						net = NeuralNet(
							layers=[
								('input', layers.InputLayer),
								('hidden1', layers.DenseLayer),
								('dropout1', layers.DropoutLayer),
								('hidden2', layers.DenseLayer),
								('dropout2', layers.DropoutLayer),
								('hidden3', layers.DenseLayer),
								('dropout3', layers.DropoutLayer),
								('hidden4', layers.DenseLayer),
								('output', layers.DenseLayer),
								],
							input_shape=(None, all_histograms[0].shape[0]),
							hidden1_num_units=500,
							dropout1_p=0.1,
							hidden2_num_units=1000,
							dropout2_p=0.2,
							hidden3_num_units=500,
							dropout3_p=0.3,
							hidden4_num_units=300,
							output_num_units=10, output_nonlinearity=nonlinearities.softmax,
							update=nesterov_momentum,
							update_learning_rate=0.001,
							update_momentum=0.9,

							eval_size=0.2,
							verbose=1,
							max_epochs=300
						)
						all_histograms_sh = []
						final_labels_sh = []
						index_shuffled = range(len(final_labels))
						shuffle(index_shuffled)
						for i in index_shuffled:
						    all_histograms_sh.append(all_histograms[i])
						    final_labels_sh.append(final_labels[i])
						all_histograms = all_histograms_sh
						final_labels = final_labels_sh

						all_histograms = np.array(all_histograms).astype(np.int32)
						final_labels = np.array(final_labels).astype(np.int32)
						net.fit(all_histograms, final_labels)
						accuracy = sum([i['valid_accuracy'] for i in net.train_history_][-50:]) / 50
						print """accuracy for k: {}, neighbors_num: {}, classifier: {} is {}%""".format(
							k, neighbors_num, classifier, accuracy*100)
						if accuracy > max_accuracy:
							best_k = k
							best_neighbors = neighbors_num
							best_kernel = kernel
							best_gamma = gamma
							max_accuracy = accuracy
							best_classifier = classifier
						sys.stdout.flush()

					elif classifier == ClassifierType.RANDOM_FOREST:
						for max_depth in [10, 60]:#[10, 30, 60, 100]:
							self.MAX_DEPTH = max_depth
							for n_estimators in [50, 200]:#[50, 200, 600]:
								self.N_ESTIMATORS = n_estimators
								for max_features in ['log2', 'auto', 'sqrt']:
									self.MAX_FEATURES = max_features

									rfc = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,
											max_features=max_features)
									scores = cross_validation.cross_val_score(rfc, all_histograms, final_labels,
										cv=cross_validation.KFold(len(final_labels), n_folds=5, shuffle=True))
									print "scores: {}".format(scores)
									accuracy = sum(scores) / len(scores)
									print """accuracy for k: {}, neighbors_num: {} max_depth: {}, n_estimators: {}, max_features: {} classifier: {} is {}%""".format(
										k, neighbors_num, max_depth, n_estimators, max_features, classifier, accuracy*100)

									if accuracy > max_accuracy:
										best_k = k
										best_neighbors = neighbors_num
										best_kernel = kernel
										best_gamma = gamma
										max_accuracy = accuracy
										best_classifier = classifier

						sys.stdout.flush()

						print "\n!!!\nbest for RandomForest: max_depth = {}, n_estimators = {}, max_features = {}".format(
						max_depth, n_estimators, max_features)
						print "max_accuracy = {}\n\n".format(max_accuracy)

					elif classifier == ClassifierType.SVM_LINEAR:
						for kernel in ['linear']:#['rbf', 'linear']:
							self.KERNEL = kernel

							if kernel == "rbf":
								for gamma in [0.01, 0.1, 0.2, 0.5, 1, 2]:
									self.GAMMA = gamma
									for C in [0.1, 0.5, 1, 2, 5]:
										svm = SVC(kernel = "rbf", gamma = self.GAMMA, C =C)
										scores = cross_validation.cross_val_score(svm, all_histograms, final_labels,
											cv=cross_validation.KFold(len(final_labels), n_folds=5, shuffle=True))
										print "scores: {}".format(scores)
										accuracy = sum(scores) / len(scores)
										print """accuracy for k: {}, neighbors_num: {}, gamma: {}, C: {}, kernel: {} classifier: {} is {}%""".format(
											k, neighbors_num, gamma, C, kernel, 'SVM', accuracy*100)

										if accuracy > max_accuracy:
											best_k = k
											best_neighbors = neighbors_num
											best_kernel = kernel
											best_gamma = gamma
											max_accuracy = accuracy
											best_classifier = classifier
								sys.stdout.flush()
							else:
								for C in [0.5, 1, 2]:#[0.1, 0.5, 1, 2, 5]:
									svm = SVC(kernel = "linear", gamma = self.GAMMA)

									scores = cross_validation.cross_val_score(svm, all_histograms, final_labels,
										cv=cross_validation.KFold(len(final_labels), n_folds=5, shuffle=True))
									print "scores: {}".format(scores)
									accuracy = sum(scores) / len(scores)
									print """accuracy for k: {}, neighbors_num: {}, C: {}, kernel: {} classifier: {} is {}%""".format(
										k, neighbors_num, C, kernel, 'SVM', accuracy*100)

									if accuracy > max_accuracy:
										best_k = k
										best_neighbors = neighbors_num
										best_kernel = kernel
										best_gamma = gamma
										max_accuracy = accuracy
										best_classifier = classifier
								sys.stdout.flush()

		print "\n!!!\nbest kernel = {}, neighbors_num = {}, k = {}, gamma = {}, classifier = {}".format(
			best_kernel, best_neighbors, best_k, gamma, best_classifier)
		print "max_accuracy = {}".format(max_accuracy)

	def show_predicted(self, test_images, labels, class_names):
		fig = plt.figure(1, figsize=(12, 5))
		fig.canvas.set_window_title('Predicted labels')
		image_no = 0
		for img in test_images:
			fig.add_subplot(3,8, image_no+1)
			plt.title(class_names[int(labels[image_no])])
			self.showfig(test_images[image_no], None)
			image_no += 1
		plt.show()

	def showfig(self, image, ucmap):
		# There is a difference in pixel ordering in OpenCV and Matplotlib.
		# OpenCV follows BGR order, while matplotlib follows RGB order.
		if len(image.shape) == 3:
			b,g,r = cv2.split(image)       # get b,g,r
			image = cv2.merge([r,g,b])     # switch it to rgb
		imgplot = plt.imshow(image, ucmap)
		imgplot.axes.get_xaxis().set_visible(False)
		imgplot.axes.get_yaxis().set_visible(False)
