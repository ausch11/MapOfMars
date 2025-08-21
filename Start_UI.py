"""MapOfMar界面的运行主程序"""

# 忽略版本不一致导致的报错
import warnings
import faulthandler

from PyQt5.QtGui import QPainter

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*sipPyTypeDict.*")

# 导入需要的库
import traceback
from Windows.MainFrame import Ui_MainWindow
from WindowClass import *  # 导入封装好的窗口
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMessageBox,
    QTableWidgetItem
)
from PyQt5.QtCore import Qt,pyqtSignal
import img_utils


class MainWin(QMainWindow, Ui_MainWindow):
    # 定义自定义信号
    classification_progress = pyqtSignal(int)  # 进度更新信号
    classification_result = pyqtSignal(object)  # 分类结果信号
    def __init__(self):
        super(MainWin, self).__init__()
        self.setupUi(self)
        # 调用监听函数
        self.controller()
        # 初始化变量
        self.openpath = None
        self.ds = None
        self.intrinsic = {}
        self.proj = {}
        self.bands = {}
        # 缓存两个窗口和元数据信息
        self.cls_window = None
        self.info_win = None
        # 显示图像的属性
        self.overlayToggle.setChecked(False)
        self.base_pixmap = None  # 原始底图
        self.current_pixmap = None  # 缓存底图
        self.overlay_data = None  # 缓存掩膜 RGBA 数组
        self.overlay_opacity = 0.6  # 初始化不透明度
        self.ShowSlider.setRange(0, 100)
        self.ShowSlider.setValue(int(self.overlay_opacity * 100))
        # 色彩索引表
        self.color_names = {
            (31, 119, 180): "曲线型沙丘",
            (44, 160, 44): "斜坡条纹",
            (127, 127, 127): "纹理",
            (140, 86, 74): "粗糙",
            (148, 103, 189): "混合",
            (152, 223, 138): "沟槽",
            (174, 199, 232): "直线型沙丘",
            (196, 156, 148): "土丘",
            (197, 176, 213): "山脊",
            (214, 39, 40): "冲沟",
            (227, 119, 194): "撞击坑群",
            (247, 182, 210): "光滑形貌",
            (255, 127, 14): "悬崖",
            (255, 152, 150): "滑坡",
            (255, 187, 120): "撞击坑"
        }


    # 监听事件都放在这里面
    def controller(self):
        self.open_action.triggered.connect(self.on_open) # 打开影像
        # 打开子窗口
        self.clsaction.triggered.connect(self.Open_ClcWindow)
        self.ImgInfo_action.triggered.connect(self.show_info)
        # 控制掩膜影像的显示
        self.overlayToggle.toggled.connect(self.toggle_overlay)
        self.ClsresultButton.clicked.connect(self.open_overlay)
        self.ShowSlider.valueChanged.connect(self.change_opacity)
        # 多线程信号（进度条、显示图像）
        self.classification_progress.connect(self.update_progress)
        self.classification_result.connect(self.handle_new_overlay)

    def update_progress(self, percent: int):
        """把 slic_map 传来的进度值更新到 GUI"""
        self.progressBar.setValue(percent)

    def handle_new_overlay(self, rgb_array: np.ndarray):
        """
        供 slic_map 调用，把新的掩膜数组设置到主窗口，
        然后立刻重绘 overlay（保留当前 transform）。
        """
        self.overlay_data = rgb_array
        # 获取颜色索引并与地貌对应
        unique_colors = img_utils.get_unique_colors(rgb_array)
        html = '<h3></h3>'
        html += '<table cellspacing="2" cellpadding="2">'
        for color in unique_colors:
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            color_tuple = (r, g, b)
            color_name = self.color_names.get(color_tuple, "未命名类别")
            hexcol = f'#{r:02X}{g:02X}{b:02X}'
            html += (
                '<tr>'
                f'  <td bgcolor="{hexcol}" width="16" height="16"></td>'
                f'  <td style="padding-left: 8px;">{color_name}</td>'
                '</tr>'
            )
        html += '</table>'

        self.legend.setAcceptRichText(True)
        self.legend.setHtml(html)
        # 如果 toggle 已选中，或者你想每次都显示：
        if not self.overlayToggle.isChecked():
            self.overlayToggle.setChecked(True)
            self.repaint_overlay()
        else:
            self.repaint_overlay()

    def on_open(self):
        """打开遥感影像文件"""
        # fp, _ = QFileDialog.getOpenFileName(self, "打开遥感影像", "",
        #                                     "影像(*.tif *.tiff *.img);;所有(*.*)")
        fp = r"E:\a_GUIRS_资料\JezeroCrater.tif";
        self.openpath = fp
        if not fp: return
        try:
            self.ds, self.intrinsic, self.proj, self.bands = img_utils.load_image_all(fp, parent=self)
        except Exception as e:
            self.statusBar().showMessage(str(e))
            return
        # 显示第一波段
        pix = img_utils.make_pixmap_from_band(self.bands[1],
                                              self.ImageView.width(),
                                              self.ImageView.height())
        self.base_pixmap = pix
        self.current_pixmap = pix
        self.ImageView.loadPixmap(pix)
        self.statusBar().showMessage(f"已加载 {fp}")

    def open_overlay(self):
        """打开图斑影像（栅格掩膜）"""
        # fp, _ = QFileDialog.getOpenFileName(
        #     self, "打开图斑影像", "",
        #     "图斑影像 (*.png);;所有文件 (*.*)"
        # )
        fp = r"E:\a_GUIRS_资料\pro_results_MRFmap_superpixelsize65_K3976_M0.2_sigma5.png"
        if not fp:
            return
        try:
            # 叠加显示
            if self.current_pixmap is None:
                QMessageBox.warning(self, "提示", "请打开原始的遥感影像")
                return

            # 使用上下文管理器确保资源释放
            ds = gdal.Open(fp, gdal.GA_ReadOnly)
            if ds is None:
                QMessageBox.critical(self, "错误", "无法打开掩膜影像")
                return
            else:
                h_mask, w_mask = ds.RasterXSize, ds.RasterYSize
                h_raw, w_raw = self.ds.RasterXSize, self.ds.RasterYSize
                if h_mask != h_raw or w_mask != w_raw:
                    QMessageBox.critical(self, "错误", "打开掩膜影像与遥感影像尺寸不一致")
                    return

            # 读取掩膜前3个波段
            bands = []
            for i in range(1, 4):
                band = ds.GetRasterBand(i)
                arr = band.ReadAsArray()
                if arr is None:
                    QMessageBox.critical(self, "错误", f"无法读取波段 {i}")
                    return
                bands.append(arr.astype(np.uint8))

            rgb = np.dstack(bands[:3])  # RGB通道
            self.overlay_data = rgb
            # 叠加显示
            self.repaint_overlay()
            # 改变radioButton的状态
            self.overlayToggle.setChecked(True)

            unique_colors = img_utils.get_unique_colors(rgb)

            html = """
            <table cellspacing="2" cellpadding="2" style="font-family:Arial, sans-serif; font-size:9pt; line-height:1.65; width:1.5;">
            """

            for color in unique_colors:
                r, g, b = int(color[0]), int(color[1]), int(color[2])
                color_tuple = (r, g, b)
                color_name = self.color_names.get(color_tuple, "未命名类别")
                hexcol = f'#{r:02X}{g:02X}{b:02X}'

                html += f"""
                <tr>
                  <td style="background-color:{hexcol}; width:40px; height:40px; border:1px solid #000;"></td>
                  <td style="padding-left:4px;">{color_name}</td>
                </tr>
                """

            html += "</table>"

            self.legend.setAcceptRichText(True)
            self.legend.setHtml(html)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理掩膜时出错: {str(e)}")
        finally:
            # 显式释放GDAL数据集
            ds = None

    def change_opacity(self, value):
        """
        根据slider设置透明度，重新绘制 overlay
        """
        self.overlay_opacity = value / 100.0
        # 如果已经有底图和掩膜，就重画一次
        if self.overlayToggle.isChecked() and getattr(self, 'overlay_data', None) is not None:
            self.repaint_overlay()

    def toggle_overlay(self, checked: bool):
        """
        判断是否显示掩膜
        """
        if checked:
            if checked:
                if self.overlay_data is None:
                    QMessageBox.information(self, "提示", "请先加载掩膜影像")
                    self.overlayToggle.setChecked(False)
                    return
                # 基于 base_pixmap 重新叠加
                self.repaint_overlay()
        else:
            iv = self.ImageView
            if hasattr(iv, 'pixItem'):
                iv.pixItem.setPixmap(self.base_pixmap)
            # current_pixmap 也重置为底图
            self.current_pixmap = self.base_pixmap

    def repaint_overlay(self):
        """在底图上叠加半透明掩膜并显示"""
        base = QPixmap(self.base_pixmap)  # 复制底图
        painter = QPainter(base)
        try:
            # 创建掩膜图像
            h, w, _ = self.overlay_data.shape
            data = np.ascontiguousarray(self.overlay_data)
            bytes_per_line = 3 * w
            mask_img = QImage(data.data, w, h, bytes_per_line, QImage.Format_RGB888)
            # 创建掩膜QPixmap
            mask_pix = QPixmap.fromImage(mask_img).scaled(
                base.size(),
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )
            # 设置半透明绘制
            painter.setOpacity(self.overlay_opacity)
            # 绘制到整个底图区域
            painter.drawPixmap(0, 0, mask_pix)

        except Exception as e:
            print("Overlay 绘制失败：", e)

        finally:
            painter.end()

        iv = self.ImageView
        if hasattr(iv, 'pixItem'):
            iv.pixItem.setPixmap(base)
        else:
            iv.loadPixmap(base)

        self.current_pixmap = base

    def show_info(self):
        """打开 Info 窗口，把 intrinsic 显示到 InfoTab1，把 proj 显示到 InfoTab2"""
        if self.info_win is None:
            self.info_win = ImgInfoWindow(parent=self)
            self.info_win.destroyed.connect(lambda: setattr(self, 'info_win', None))

        InfoTab1, InfoTab2 = self.info_win.InfoTab1, self.info_win.InfoTab2

        # 填 InfoTab1
        for row, key in enumerate(self.info_win.intrinsic_keys):
            val = self.intrinsic.get(key, "")
            InfoTab1.setItem(row, 1, QTableWidgetItem(str(val)))

        # 填 InfoTab2
        for row, key in enumerate(self.info_win.proj_keys):
            val = self.proj.get(key, "")
            InfoTab2.setItem(row, 1, QTableWidgetItem(str(val)))

        if self.info_win.isMinimized():
            self.info_win.showNormal()
        else:
            self.info_win.show()
        self.info_win.activateWindow()

    def Open_ClcWindow(self):
        """打开分类窗口，使用单例模式确保只打开一个"""
        if self.cls_window is None:
            # 创建新窗口，并设置主窗口为父对象
            self.cls_window = ClassWindow(parent=self,
                                          progress_signal=self.classification_progress,
                                          result_signal=self.classification_result)
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
# 输出底层出现问题时候报错
faulthandler.enable(all_threads=True)

if __name__ == '__main__':
    app = QApplication(sys.argv)  # application 对象
    main_window = MainWin()  # QMainWindow对象
    main_window.show()  # 显示
    sys.exit(app.exec_())  # 启动Qt事件循环, 并确保程序退出时返回正确状态码
