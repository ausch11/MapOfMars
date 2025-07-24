"""MapOfMar界面的运行主程序"""

# 忽略版本不一致导致的报错
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*sipPyTypeDict.*")

# 导入需要的库
import traceback
from Windows.MainFrame import Ui_MainWindow
from WindowClass import * # 导入封装好的窗口
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox,
    QTabWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt
import img_utils


class MainWin(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWin, self).__init__()
        self.setupUi(self)
        # 调用监听函数
        self.controller()
        # 初始化变量
        self.ds = None
        self.intrinsic = {}
        self.proj = {}
        self.bands = {}
        # 缓存两个窗口和元数据信息
        self.cls_window = None
        self.info_win = None

    # 监听事件都放在这里面
    def controller(self):
        self.open_action.triggered.connect(self.on_open)
        self.clsaction.triggered.connect(self.Open_ClcWindow)
        self.ImgInfo_action.triggered.connect(self.show_info)
        self.testButton.clicked.connect(self.on_open)

    def on_open(self):
        """打开遥感影像文件"""
        fp, _ = QFileDialog.getOpenFileName(self, "打开遥感影像", "",
                                            "影像(*.tif *.tiff *.img);;所有(*.*)")
        if not fp: return
        try:
            self.ds, self.intrinsic, self.proj, self.bands = img_utils.load_image_all(fp, parent=self)
        except Exception as e:
            self.statusBar().showMessage(str(e))
            return
        # 显示第一波段
        pix = img_utils.make_pixmap_from_band(self.bands[1],
                                                  self.imglabel.width(),
                                                  self.imglabel.height())
        self.imglabel.setPixmap(pix)
        self.statusBar().showMessage(f"已加载 {fp}")

    def show_info(self):
        """打开 Info 窗口，把 intrinsic 显示到 InfoTab1，把 proj 显示到 InfoTab2"""
        if self.info_win is None:
            self.info_win = ImgInfoWindow(parent=self)
            self.info_win.destroyed.connect(lambda: setattr(self, 'info_win', None))

        InfoTab1, InfoTab2 = self.info_win.InfoTab1, self.info_win.InfoTab2

        # 填 InfoTab1
        InfoTab1.clear()
        InfoTab1.setRowCount(len(self.intrinsic))
        InfoTab1.setColumnCount(2)
        InfoTab1.setHorizontalHeaderLabels(["属性", "值"])
        for i, (k, v) in enumerate(self.intrinsic.items()):
            InfoTab1.setItem(i, 0, QTableWidgetItem(k))
            InfoTab1.setItem(i, 1, QTableWidgetItem(str(v)))
        InfoTab1.resizeColumnsToContents()

        # 填 InfoTab2
        InfoTab2.clear()
        InfoTab2.setRowCount(len(self.proj))
        InfoTab2.setColumnCount(2)
        InfoTab2.setHorizontalHeaderLabels(["参数", "数值"])
        for i, (k, v) in enumerate(self.proj.items()):
            InfoTab2.setItem(i, 0, QTableWidgetItem(k))
            InfoTab2.setItem(i, 1, QTableWidgetItem(str(v)))
        InfoTab2.resizeColumnsToContents()

        self.info_win.show()
        self.info_win.activateWindow()

    def Open_ClcWindow(self):
        """打开分类窗口，使用单例模式确保只打开一个"""
        if self.cls_window is None:
            # 创建新窗口，并设置主窗口为父对象
            self.cls_window = ClassWindow(parent=self)
            # 当窗口关闭时自动清除引用
            self.cls_window.destroyed.connect(lambda: setattr(self, 'cls_window', None))

        # 设置为模态窗口
        self.cls_window.setWindowModality(Qt.ApplicationModal)
        self.cls_window.show()
        self.cls_window.activateWindow()

    def open_ImgInfoWindow(self):
        if self.imginfo_window is None:
            self.imginfo_window = ImgInfoWindow(parent=self)
            self.imginfo_window.destroyed.connect(
                lambda: setattr(self, 'imginfo_window', None)
            )

        InfoTab1 = self.imginfo_window.InfoTab1
        InfoTab2 = self.imginfo_window.InfoTab2

        # 填充 InfoTab1（Intrinsic）
        data1 = self.current_intrinsic
        InfoTab1.clear()
        InfoTab1.setRowCount(len(data1))
        InfoTab1.setColumnCount(2)
        InfoTab1.setHorizontalHeaderLabels(["属性", "值"])
        for row, (k, v) in enumerate(data1.items()):
            InfoTab1.setItem(row, 0, QTableWidgetItem(k))
            InfoTab1.setItem(row, 1, QTableWidgetItem(str(v)))
        InfoTab1.resizeColumnsToContents()

        # 填充 InfoTab2（Projection）
        data2 = self.current_proj
        InfoTab2.clear()
        InfoTab2.setRowCount(len(data2))
        InfoTab2.setColumnCount(2)
        InfoTab2.setHorizontalHeaderLabels(["参数", "数值"])
        for row, (k, v) in enumerate(data2.items()):
            InfoTab2.setItem(row, 0, QTableWidgetItem(k))
            InfoTab2.setItem(row, 1, QTableWidgetItem(str(v)))
        InfoTab2.resizeColumnsToContents()

        # 最后显示
        if self.imginfo_window.isMinimized():
            self.imginfo_window.showNormal()
        else:
            self.imginfo_window.show()
        self.imginfo_window.activateWindow()


# 显示QT运行中存在的错误
def my_excepthook(exc_type, exc_value, exc_tb):
    """全局未捕获异常处理器，打印到控制台并弹窗显示。"""
    tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(tb, file=sys.stderr)  # 在命令行里也能看到
    # 如果你想弹窗：
    QMessageBox.critical(None, "错误信息", tb)
    # 结束进程（可选）
    sys.exit(1)


# 把默认的钩子替换掉
sys.excepthook = my_excepthook

if __name__ == '__main__':
    app = QApplication(sys.argv)  # application 对象
    main_window = MainWin()  # QMainWindow对象
    main_window.show()  # 显示
    sys.exit(app.exec_())  # 启动Qt事件循环, 并确保程序退出时返回正确状态码
