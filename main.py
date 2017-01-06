import sys

from PyQt4 import QtGui

from main_window import MainWindow


def main():
	print 'Hey!'
	app = QtGui.QApplication(sys.argv)
	main_window = MainWindow()
	sys.exit(app.exec_())

if __name__ == "__main__":
    main()
