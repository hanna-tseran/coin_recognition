class DetectionMethod(object):
    WATERSHED = 'Watershed'
    HOUGH_TRANSFORM = 'Hough transform'
    CANNY = 'Canny'
    ALL = 'Combination'

class ClassifierType(object):
	SVM_LINEAR = 'SVM: linear kernel'
	SVM_RBF = 'SVM: rbf kernel'
	RANDOM_FOREST = 'Random forest'
	NEURAL_NETWORK = 'Neural network'
