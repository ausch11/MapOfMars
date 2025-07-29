from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import Qt
from Windows.class_window import Ui_ClassWindow
from Windows.ImgInfo_window import Ui_ImgInfoWindow
from PyQt5.QtWidgets import QMainWindow, QHeaderView, QTableWidgetItem

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

        self.intrinsic_keys = [
            "尺寸(像素)", "波段数", "数据类型", "分辨率",
            "左上角坐标", "NoData 值", "驱动/格式", "存储组织"
        ]
        self.proj_keys = [
            "投影类型", "标准纬线", "中央经线",
            "假东距", "假北距", "线性单位", "椭球长半轴"
        ]
        # 初始化 InfoTab1
        t1 = self.InfoTab1
        t1.setRowCount(len(self.intrinsic_keys))
        t1.setColumnCount(2)
        t1.setHorizontalHeaderLabels(["属性", "值"])
        for i, key in enumerate(self.intrinsic_keys):
            t1.setItem(i, 0, QTableWidgetItem(key))
        # 均匀拉伸两列
        header1 = t1.horizontalHeader()
        header1.setSectionResizeMode(0, QHeaderView.Stretch)
        header1.setSectionResizeMode(1, QHeaderView.Stretch)

        # 初始化 InfoTab2
        t2 = self.InfoTab2
        t2.setRowCount(len(self.proj_keys))
        t2.setColumnCount(2)
        t2.setHorizontalHeaderLabels(["参数", "数值"])
        for i, key in enumerate(self.proj_keys):
            t2.setItem(i, 0, QTableWidgetItem(key))
        # 同样均匀拉伸
        header2 = t2.horizontalHeader()
        header2.setSectionResizeMode(0, QHeaderView.Stretch)
        header2.setSectionResizeMode(1, QHeaderView.Stretch)

    def closeEvent(self, event):
        """重写关闭事件，确保完全关闭而不是隐藏"""
        self.deleteLater()
        super().closeEvent(event)