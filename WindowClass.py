from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt
from Windows.class_window import Ui_ClassWindow
from Windows.ImgInfo_window import Ui_ImgInfoWindow

class ClassWindow(QMainWindow, Ui_ClassWindow):
    def __init__(self, parent=None):
        super(ClassWindow, self).__init__(parent)
        self.setupUi(self)
        # 设置窗口标志，使其作为独立窗口但保持与主窗口的关系
        self.setWindowFlag(Qt.Window, True)

    def closeEvent(self, event):
        """重写关闭事件，确保完全关闭而不是隐藏"""
        self.deleteLater()
        super().closeEvent(event)


class ImgInfoWindow(QMainWindow, Ui_ImgInfoWindow):
    def __init__(self, parent=None):
        super(ImgInfoWindow, self).__init__(parent)
        self.setupUi(self)

    def closeEvent(self, event):
        """重写关闭事件，确保完全关闭而不是隐藏"""
        self.deleteLater()
        super().closeEvent(event)