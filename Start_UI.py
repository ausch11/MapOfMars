"""MapOfMar界面的运行主程序"""

# 忽略版本不一致导致的报错
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*sipPyTypeDict.*")

# 导入需要的库
import traceback
from MapOfMars.MainFrame import Ui_MainWindow
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from img_show import ImageProcessor
from class_window import Ui_ClassWindow
from ImgInfo_window import Ui_ImgInfowindow


class MainWin(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWin, self).__init__()
        self.setupUi(self)
        # 调用监听函数
        self.controller()
        # 初始化变量
        self.image_path = ""
        self.dataset = None
        self.band_data = {}
        self.current_band = 1
        self.cls_window = None  # 存储分类窗口的引用
        self.imginfo_window = None

    # 监听事件都放在这里面
    def controller(self):
        self.open_action.triggered.connect(self.Select_Image)
        self.clsaction.triggered.connect(self.Open_ClcWindow)
        self.ImgInfo_action.triggered.connect(self.open_ImgInfoWindow)

    def Select_Image(self):
        """打开遥感影像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开遥感影像", "",
            "遥感影像 (*.tif *.tiff *.img *.jp2 *.hdf);;所有文件 (*.*)"
        )

        if file_path:
            self.image_path = file_path
            self.load_image()

    def load_image(self):
        """加载并处理遥感影像"""
        # 使用工具类处理图像
        self.dataset, self.band_data, info, error = ImageProcessor.load_image(
            self.image_path,
            self.statusbar,
            parent=self
        )

        if self.dataset and self.band_data:
            # 显示第一个波段
            self.current_band = 1
            self.update_image()
            self.textEdit.setPlainText(info)
        elif error:
            self.statusbar.showMessage(f"加载失败: {error}")

    def update_image(self):
        """更新显示的图像"""
        pixmap = ImageProcessor.update_image(
            self.band_data,
            self.current_band,
            self.label,
            self.statusbar
        )

        if pixmap:
            self.label.setPixmap(pixmap)

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
            # 创建新窗口，并设置主窗口为父对象
            self.imginfo_window = ImgInfoWindow(parent=self)
            # 当窗口关闭时自动清除引用
            self.imginfo_window.destroyed.connect(lambda: setattr(self, 'cls_window', None))

        # 如果窗口已最小化则恢复，否则激活
        if self.imginfo_window.isMinimized():
            self.imginfo_window.showNormal()
        else:
            self.imginfo_window.show()
            self.imginfo_window.activateWindow()


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


class ImgInfoWindow(QMainWindow, Ui_ImgInfowindow):
    def __init__(self, parent=None):
        super(ImgInfoWindow, self).__init__(parent)
        self.setupUi(self)
        # 设置窗口标志，使其作为独立窗口但保持与主窗口的关系
        self.setWindowFlag(Qt.Window, True)

    def closeEvent(self, event):
        """重写关闭事件，确保完全关闭而不是隐藏"""
        self.deleteLater()
        super().closeEvent(event)


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
