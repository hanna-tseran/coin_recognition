import cv2

from PyQt4 import QtCore
from PyQt4 import QtGui

from detection import Detection
from declarations import DetectionMethod, ClassifierType
from categorization import Categorization
from detection_quality import DetectionQuality


class MainWindow(QtGui.QMainWindow):

    central_widget = None
    picture_label = None

    img_filename = 'test_images/coins_main.jpg'
    detection_method = DetectionMethod.ALL
    classifier_type = ClassifierType.NEURAL_NETWORK
    ctn = Categorization()

    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.central_widget = QtGui.QWidget()

        img = self.convertToQImage(cv2.imread(self.img_filename))
        pixmap = QtGui.QPixmap(img)
        self.picture_label = QtGui.QLabel(self)
        self.picture_label.setPixmap(pixmap)
        self.picture_label.resize(self.picture_label.sizeHint())

        detectButton = QtGui.QPushButton('Detect coins', self)
        detectButton.resize(detectButton.sizeHint())
        detectButton.clicked.connect(self.detectButtonClicked)

        uploadButton = QtGui.QPushButton('Select image', self)
        detectButton.resize(uploadButton.sizeHint())
        uploadButton.clicked.connect(self.uploadButtonClicked)

        detectionCombo = QtGui.QComboBox(self)
        detectionCombo.addItem(DetectionMethod.ALL)
        detectionCombo.addItem(DetectionMethod.WATERSHED)
        detectionCombo.addItem(DetectionMethod.HOUGH_TRANSFORM)
        detectionCombo.addItem(DetectionMethod.CANNY)
        detectionCombo.activated[str].connect(self.detectionComboActivated)

        classifierCombo = QtGui.QComboBox(self)
        classifierCombo.addItem(ClassifierType.NEURAL_NETWORK)
        classifierCombo.addItem(ClassifierType.SVM_LINEAR)
        classifierCombo.addItem(ClassifierType.RANDOM_FOREST)
        classifierCombo.activated[str].connect(self.classifierComboActivated)

        categorizeButton = QtGui.QPushButton('Categorize coins', self)
        categorizeButton.resize(categorizeButton.sizeHint())
        categorizeButton.clicked.connect(self.categorizeButtonClicked)

        buttons_widget = QtGui.QWidget()
        buttons_box = QtGui.QHBoxLayout()
        buttons_box.addStretch(1)
        buttons_box.addWidget(uploadButton)
        buttons_box.addWidget(detectionCombo)
        buttons_box.addWidget(detectButton)
        buttons_box.addWidget(classifierCombo)
        buttons_box.addWidget(categorizeButton)
        buttons_widget.setLayout(buttons_box)

        gridLayout = QtGui.QGridLayout()
        gridLayout.addWidget(self.picture_label,0,0,QtCore.Qt.AlignCenter)
        gridLayout.addWidget(buttons_widget,1,0)
        self.central_widget.setLayout(gridLayout)
        self.setCentralWidget(self.central_widget)

        self.statusBar()

        self.center()
        self.setWindowTitle('Coin Recognition')
        self.setWindowIcon(QtGui.QIcon('icon.jpg'))
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.resize(self.sizeHint())
        self.setFixedSize(self.size())
        self.show()

    def detectButtonClicked(self):
        detection = Detection()
        new_pixmap = QtGui.QPixmap(self.convertToQImage(
            detection.detect_coins(self.img_filename, self.detection_method)))
        self.picture_label.setPixmap(new_pixmap)
        self.statusBar().showMessage('Detection method: {}'.format(self.detection_method))

    def uploadButtonClicked(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '.')
        self.img_filename = str(filename)
        new_pixmap = QtGui.QPixmap(self.convertToQImage(cv2.imread(str(filename))))
        self.picture_label.setPixmap(new_pixmap)

        self.picture_label.resize(self.picture_label.sizeHint())
        self.central_widget.resize(self.central_widget.sizeHint())
        self.setMinimumSize(self.sizeHint())
        self.resize(self.sizeHint())

    def detectionQualityButtonClicked(self):
        detection_quality = DetectionQuality()
        detection_quality.measure_quality(self.detection_method)

    def detectionComboActivated(self, text):
        if text == DetectionMethod.WATERSHED:
            self.detection_method = DetectionMethod.WATERSHED
        elif text == DetectionMethod.HOUGH_TRANSFORM:
            self.detection_method = DetectionMethod.HOUGH_TRANSFORM
        elif text == DetectionMethod.CANNY:
            self.detection_method = DetectionMethod.CANNY
        elif text == DetectionMethod.ALL:
            self.detection_method = DetectionMethod.ALL

    def classifierComboActivated(self, text):
        self.classifier_type = text

    def categorizeButtonClicked(self):
        self.ctn.categorize_coins(self.img_filename, self.detection_method, self.classifier_type)

    def convertToQImage(self, img):
        height, width, bytesPerComponent = img.shape
        bytesPerLine = bytesPerComponent * width;
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

    def center(self):
        frame = self.frameGeometry()
        window_center = QtGui.QDesktopWidget().availableGeometry().center()
        frame.moveCenter(window_center)
        self.move(frame.topLeft())
