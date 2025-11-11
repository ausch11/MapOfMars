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
    QTableWidgetItem, QLabel
)
from PyQt5.QtCore import Qt,pyqtSignal
import img_utils
import os


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
        self.geotransform = None  # 地理坐标转换相关变量
        self.projection = None
        self.coord_label = None  # 状态栏坐标标签
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
        # 连接裁切信号
        self.ImageView.cropped.connect(self.on_cropped)
        # 初始化状态栏坐标显示
        self.init_coord_display()

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
        self.open_action.triggered.connect(self.on_open)  # 打开影像
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
        # 关闭图像or掩膜
        self.clos_imgaction.triggered.connect(self.close_image)
        self.clos_ovrButton.clicked.connect(self.close_overlay)

    def init_coord_display(self):
        """初始化状态栏坐标显示（右侧固定显示），初始隐藏，只有有影像时才显示"""
        # 创建一个右对齐的 QLabel 放到状态栏的永久区域
        self.coord_label = QLabel("")
        self.coord_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # 可调的最小宽度，保证信息显示完整；按需修改
        self.coord_label.setMinimumWidth(220)
        # 给点内边距让文字不贴边
        self.coord_label.setStyleSheet("padding-right:8px;")
        # 初始隐藏（未加载影像时不显示）
        self.coord_label.hide()
        # 将其添加到状态栏的永久区域（默认会靠右）
        self.statusBar().addPermanentWidget(self.coord_label, 0)

        # 连接鼠标移动信号（保持你原来的信号连接）
        self.ImageView.mouseMoveSignal.connect(self.update_coord_display)

    def _ensure_georef(self):
        """
        尝试从 self.geotransform/self.projection（优先）或 self.intrinsic/self.proj 中提取地理信息，
        以避免重复从文件读取。支持常见键名的回退。
        """
        # geotransform
        if self.geotransform is None:
            # 常见键名回退尝试
            possible_gt = (
                self.intrinsic.get('geotransform') if isinstance(self.intrinsic, dict) else None,
                self.intrinsic.get('GeoTransform') if isinstance(self.intrinsic, dict) else None,
                self.intrinsic.get('gt') if isinstance(self.intrinsic, dict) else None,
                self.intrinsic.get('transform') if isinstance(self.intrinsic, dict) else None,
            )
            for gt in possible_gt:
                if gt:
                    # 有的可能已经是 tuple/list
                    try:
                        self.geotransform = tuple(gt)
                        break
                    except Exception:
                        # 如果是单字符串需要解析（不做复杂解析）
                        pass

        # projection (WKT)
        if self.projection is None:
            possible_pr = (
                self.proj.get('wkt') if isinstance(self.proj, dict) else None,
                self.proj.get('projection') if isinstance(self.proj, dict) else None,
                self.proj.get('proj_wkt') if isinstance(self.proj, dict) else None,
                self.proj.get('wkt_string') if isinstance(self.proj, dict) else None,
                self.proj.get('WKT') if isinstance(self.proj, dict) else None,
            )
            for pr in possible_pr:
                if pr:
                    self.projection = pr
                    break

    def update_coord_display(self, x, y):
        """更新右侧坐标标签（如果未加载影像则隐藏并返回）"""
        # 如果没有加载影像（既没有 ds，也没有从 load_image_all 填充的 intrinsic/proj），则隐藏坐标并返回
        no_image_loaded = (getattr(self, 'ds', None) is None) and (not bool(self.intrinsic))
        if no_image_loaded:
            if hasattr(self, "coord_label") and self.coord_label.isVisible():
                self.coord_label.hide()
            return

        # 确保在有影像时标签可见
        if hasattr(self, "coord_label") and not self.coord_label.isVisible():
            self.coord_label.show()

        # 尝试确保 geotransform / projection 可用（从 intrinsic/proj 回退）
        self._ensure_georef()

        # 计算地理坐标（如果可以），否则显示像素坐标
        if getattr(self, "geotransform", None) is not None:
            gt = self.geotransform
            try:
                geo_x = gt[0] + x * gt[1] + y * gt[2]
                geo_y = gt[3] + x * gt[4] + y * gt[5]
            except Exception:
                # 若 geotransform 格式异常，退回显示像素
                text = f"像素坐标: X: {x:.4f}, Y: {y:.4f}"
                self.coord_label.setText(text)
                return

            if getattr(self, "projection", None):
                try:
                    source = osr.SpatialReference()
                    # projection 可能已经是 WKT 字符串或其它
                    source.ImportFromWkt(self.projection)
                    target = osr.SpatialReference()
                    target.ImportFromEPSG(4326)  # WGS84
                    transform = osr.CoordinateTransformation(source, target)
                    lon, lat, _ = transform.TransformPoint(geo_x, geo_y)
                    text = f"经度: {lon:.6f}°, 纬度: {lat:.6f}°"
                except Exception:
                    # 投影转换失败时显示投影坐标（地理坐标系下的 X/Y）
                    text = f"X: {geo_x:.2f}, Y: {geo_y:.2f}"
            else:
                text = f"X: {geo_x:.2f}, Y: {geo_y:.2f}"
        else:
            # 没有地理参考信息，显示像素坐标（但仅在已加载影像时）
            text = f"像素坐标: X: {x:.4f}, Y: {y:.4f}"

        # 更新标签文本
        if hasattr(self, "coord_label") and self.coord_label is not None:
            self.coord_label.setText(text)
        else:
            self.statusBar().showMessage(text)

    def close_image(self):
        """关闭当前打开的影像"""
        if self.ds is None:
            QMessageBox.information(self, "提示", "没有打开的影像")
            return

        # 确认对话框
        reply = QMessageBox.question(self, "确认", "确定要关闭当前影像吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 清除影像相关数据
            self.openpath = None
            self.ds = None
            self.intrinsic = {}
            self.proj = {}
            self.bands = {}
            self.geotransform = None
            self.projection = None

            # 清除显示
            self.base_pixmap = None
            self.current_pixmap = None
            self.ImageView.scene().clear()
            self.ImageView._orig_pixmap = None
            self.ImageView.pixItem = None

            # 清除掩膜数据
            self.close_overlay(show_message=False)

            # 更新状态栏
            self.statusBar().showMessage("影像已关闭")
            self.coord_label.setText("")  # 清空坐标显示
        else:
            return

    def close_overlay(self, show_message=True):
        """关闭当前掩膜，保持当前视图状态"""
        if self.overlay_data is None:
            if show_message:
                QMessageBox.information(self, "提示", "没有打开的掩膜")
            return

        # 确认对话框
        if show_message:
            reply = QMessageBox.question(self, "确认", "确定要关闭当前掩膜吗？",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        # 保存当前的视图变换状态
        current_transform = None
        if hasattr(self.ImageView, 'transform') and self.ImageView.transform():
            current_transform = self.ImageView.transform()

        # 清除掩膜数据
        self.overlay_data = None
        self.ImageView.set_overlay_data(None)

        # 重置显示，但保持当前视图状态
        if self.base_pixmap is not None:
            # 直接设置底图，不调用 loadPixmap
            if hasattr(self.ImageView, 'pixItem') and self.ImageView.pixItem is not None:
                self.ImageView.pixItem.setPixmap(self.base_pixmap)

            # 恢复之前的视图变换
            if current_transform:
                self.ImageView.setTransform(current_transform)

        # 更新UI状态
        self.overlayToggle.setChecked(False)
        self.legend.clear()  # 清除图例

        if show_message:
            self.statusBar().showMessage("掩膜已关闭")

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
        # 设置显示掩膜颜色表的格式
        html = img_utils.generate_color_legend(unique_colors,self.color_names)# 其他设置参数见文档
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
        # pix = img_utils.make_pixmap_from_band(self.bands[1],
        #                                       self.ImageView.width(),
        #                                       self.ImageView.height())
        pix = img_utils.make_pixmap_from_band(self.bands[1])
        self.base_pixmap = pix
        self.current_pixmap = pix
        self.ImageView.loadPixmap(pix)
        # 清空之前的掩膜数据
        self.ImageView.set_overlay_data(None)
        self.statusBar().showMessage(f"已加载 {fp}")

    def on_cropped(self, orig_image, mask_image, overlay_image, base_name):
        """处理裁剪结果"""
        # 选择保存文件夹
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存文件夹", "")
        if not save_dir:
            return

            # 保存原始图像
        orig_path = os.path.join(save_dir, f"{base_name}_original.png")
        success_orig = orig_image.save(orig_path)

        # 保存掩膜图像（如果存在）
        success_mask = True
        if mask_image is not None:
            mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
            success_mask = mask_image.save(mask_path)

        # 保存叠加图像（如果存在）
        success_overlay = True
        if overlay_image is not None:
            overlay_path = os.path.join(save_dir, f"{base_name}_overlay.png")
            success_overlay = overlay_image.save(overlay_path)

        # 显示保存结果
        if success_orig and success_mask and success_overlay:
            QMessageBox.information(self, "保存成功",
                                    f"已保存到：\n{orig_path}" +
                                    (f"\n{mask_path}" if mask_image is not None else "") +
                                    (f"\n{overlay_path}" if overlay_image is not None else ""))
        else:
            failed_files = []
            if not success_orig:
                failed_files.append(orig_path)
            if not success_mask and mask_image is not None:
                failed_files.append(mask_path)
            if not success_overlay and overlay_image is not None:
                failed_files.append(overlay_path)
            QMessageBox.critical(self, "保存失败",
                                 f"无法保存以下文件：\n" + "\n".join(failed_files))

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
            # 将掩膜数据传递给ImageView
            self.ImageView.set_overlay_data(rgb)
            # 叠加显示
            self.repaint_overlay()
            # 改变radioButton的状态
            self.overlayToggle.setChecked(True)

            unique_colors = img_utils.get_unique_colors(rgb)

            html = img_utils.generate_color_legend(unique_colors,self.color_names)# 其他设置参数见文档

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
            if self.overlay_data is None:
                QMessageBox.information(self, "提示", "请先加载掩膜影像")
                self.overlayToggle.setChecked(False)
                return
            # 基于 base_pixmap 重新叠加
            self.repaint_overlay()
        else:
            # 保存当前的视图变换状态
            current_transform = None
            if hasattr(self.ImageView, 'transform') and self.ImageView.transform():
                current_transform = self.ImageView.transform()

            # 直接设置底图，不重置视图
            if hasattr(self.ImageView, 'pixItem') and self.ImageView.pixItem is not None:
                if self.base_pixmap is not None:
                    self.ImageView.pixItem.setPixmap(self.base_pixmap)

            # 恢复之前的视图变换
            if current_transform:
                self.ImageView.setTransform(current_transform)

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
