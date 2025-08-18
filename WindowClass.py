from Windows.class_window import Ui_ClassWindow
from Windows.ImgInfo_window import Ui_ImgInfoWindow
from PyQt5.QtWidgets import (QMainWindow, QHeaderView, QTableWidgetItem,
                             QFileDialog)
from img_utils import *
from seg_map_func import image_seg
from pathlib import Path

class ClassWindow(QMainWindow, Ui_ClassWindow):
    def __init__(self, parent=None):
        super(ClassWindow, self).__init__(parent)
        self.setupUi(self)
        # 设置窗口标志，使其作为独立窗口但保持与主窗口的关系

        self.setWindowFlag(Qt.Window, True)
        self.processButton.clicked.connect(self.proc_clicked)
        self.saveButton.clicked.connect(self.set_savepath)
        self.classification_thread = None

    def proc_clicked(self):
        # 这里既是按钮的响应函数，也能调用主窗口变量
        main = self.parent()
        fp = main.openpath
        savepath = self.p_sp.text()
        if fp is None:
            QMessageBox.critical(self, "错误", "请打开分类影像")
            return
        if savepath is None or savepath == "":
            QMessageBox.critical(self, "错误", "请选择保存路径")
            return

        type = self.p_type.currentText()
        superpixelsize = get_numeric_value(self.p_super, error_msg="超像素大小请输入整数")
        M = get_numeric_value(self.p_m, type_=float, error_msg="紧凑度请输入小数")
        windowlsize = get_numeric_value(self.p_winsize, error_msg="制图窗口大小请输入整数")
        if superpixelsize is None or M is None or windowlsize is None:
            return  # 验证不通过，立刻退出，让用户改输入

        image_seg.slic_map_pipeline(
            fp,
            savepath,
            type,
            superpixelsize,
            windowlsize,
            M,
            progress_callback=main.update_progress,
            overlay_callback=main.handle_new_overlay)

        QMessageBox.information(self, "提示", "地貌识别制图已完成")

    def set_savepath(self):
        # 弹出文件夹对话框，返回所选文件夹的路径
        directory = QFileDialog.getExistingDirectory(self, "请选择文件夹", "")
        if directory:
            # 将路径显示到 QTextEdit 中
            self.p_sp.setText(directory)
        else:
            self.p_sp = ""

    def closeEvent(self, event):
        """重写关闭事件，确保完全关闭而不是隐藏"""
        # self.deleteLater()
        # super().closeEvent(event)
        main = self.parent()
        if main is not None and hasattr(main, "update_progress"):
            main.update_progress(0)
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