import glob

from PyQt5 import QtWidgets

from Windows.class_window import Ui_ClassWindow
from Windows.ImgInfo_window import Ui_ImgInfoWindow
from Windows.Preproc_window import Ui_PreprossWin
from Windows.SampShow_window import Ui_SampleShowWin
from Windows.help_window import Ui_HelpWindow
from PyQt5.QtWidgets import (QMainWindow, QHeaderView, QTableWidgetItem,
                             QFileDialog, QGridLayout, QLabel)
from img_utils import *
import os


# 加入模型选择时候的分类窗口
class ClassWindow(QMainWindow, Ui_ClassWindow):
    def __init__(self, parent, progress_signal, result_signal):
        super(ClassWindow, self).__init__(parent)
        self.setupUi(self)
        # 设置窗口标志，使其作为独立窗口但保持与主窗口的关系
        self.progress_signal = progress_signal  # 主窗口的进度信号
        self.result_signal = result_signal  # 主窗口的结果信号
        self.classification_thread = None

        self.setWindowFlag(Qt.Window, True)
        self.processButton.clicked.connect(self.proc_clicked)
        self.saveButton.clicked.connect(self.set_savepath)

    def proc_clicked(self):
        """
        这里既是按钮的响应函数，也能调用主窗口变量
        """
        # 获取参数变量
        main = self.parent()
        fp = main.openpath
        savepath = self.p_sp.text()
        if fp is None:
            QMessageBox.critical(self, "错误", "请打开分类影像")
            return
        if savepath is None or savepath == "":
            QMessageBox.critical(self, "错误", "请选择保存路径")
            return

        # 超像素 / M / window size
        model_name = self.p_netcls.currentText()
        superpixelsize = get_numeric_value(self.p_super, error_msg="超像素大小请输入整数")
        M = get_numeric_value(self.p_m, type_=float, error_msg="紧凑度请输入小数")
        windowlsize = get_numeric_value(self.p_winsize, error_msg="制图窗口大小请输入整数")
        if superpixelsize is None or M is None or windowlsize is None:
            return  # 退出，让用户改输入


        # 从 image_seg.MODEL_REGISTRY 获取 model_spec（注意 image_seg 在你的代码中即整合后的模块）
        try:
            from seg_map_func import image_seg
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法导入 image_seg 模块: {e}")
            return

        registry = getattr(image_seg, 'MODEL_REGISTRY', None)
        if registry is None:
            QMessageBox.critical(self, "错误", " 未找到模型注册表MODEL_REGISTRY。")
            return

        if model_name not in registry:
            # 提示并列出可选项
            available = ", ".join(list(registry.keys()))
            QMessageBox.critical(self, "错误", f"选择的模型 '{model_name}' 未在注册表中找到。\n可用模型列表: {available}")
            return

        # 构建 model_spec（registry[model_name] 是一个返回 model_spec 的函数）
        try:
            model_spec = registry[model_name]()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"构建 model_spec 失败: {e}")
            return

        # 更新UI状态
        self.processButton.setEnabled(False)
        self.saveButton.setEnabled(False)

        # 创建并启动分类线程（把 model_spec 传入）
        self.classification_thread = image_seg.ClassificationThread(
            openpath=fp,
            savepath=savepath,
            superpixelsize=superpixelsize,
            window_size=windowlsize,
            M=M,
            model_spec=model_spec,
        )

        self.set_controls_enabled(False)

        # 连接信号
        self.classification_thread.progress_updated.connect(self.handle_progress_update)
        self.classification_thread.result_ready.connect(self.handle_classification_result)
        self.classification_thread.error_occurred.connect(self.handle_classification_error)
        self.classification_thread.task_completed.connect(self.handle_task_completed)
        self.classification_thread.finished.connect(self.on_thread_finished)

        # 启动线程
        self.classification_thread.start()

    def set_controls_enabled(self, enabled):
        """设置所有参数控件的启用状态"""
        # 禁用/启用参数输入控件
        self.p_netcls.setEnabled(enabled)
        self.p_super.setEnabled(enabled)
        self.p_m.setEnabled(enabled)
        self.p_winsize.setEnabled(enabled)

        # 保存路径控件
        self.p_sp.setEnabled(enabled)

        # 按钮状态
        self.processButton.setEnabled(enabled)
        self.saveButton.setEnabled(enabled)

    def handle_progress_update(self, progress):
        """处理进度更新"""
        # 转发进度到主窗口
        self.progress_signal.emit(progress)

    def handle_classification_result(self, image_data):
        """处理分类结果"""
        # 转发结果到主窗口
        self.result_signal.emit(image_data)


    def handle_classification_error(self, error_message):
        """处理分类错误"""
        QMessageBox.critical(self, "分类错误", error_message)
        self.reset_ui()

    def handle_task_completed(self):
        """处理任务完成"""
        QMessageBox.information(self, "完成", "分类任务成功完成")

    def on_thread_finished(self):
        """线程完成处理"""
        self.reset_ui()

    def reset_ui(self):
        """重置UI状态"""
        self.set_controls_enabled(True)

    def set_savepath(self):
        # 弹出文件夹对话框，返回所选文件夹的路径
        directory = QFileDialog.getExistingDirectory(self, "请选择文件夹", "")
        if directory:
            # 将路径显示到 QTextEdit 中
            self.p_sp.setText(directory)
        else:
            # 清空保存路径显示
            self.p_sp.setText("")

    def closeEvent(self, event):
        """重写关闭事件，确保完全关闭而不是隐藏"""
        main = self.parent()
        if main is not None and hasattr(main, "update_progress"):
            main.update_progress(0)
        super().closeEvent(event)


# 图像地理信息显示窗口
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


# 图像预处理窗口
class PreProcWindow(QMainWindow, Ui_PreprossWin):
    def __init__(self, parent=None):
        super(PreProcWindow, self).__init__(parent)
        self.setupUi(self)
        # 初始化默认值
        self.lineEdit.setText("0.5")  # 默认截断百分比
        # 绑定按钮事件
        self.pushButton1.clicked.connect(self.set_savepath)
        self.pushButton2.clicked.connect(self.run_gray_process)

    def set_savepath(self):
        # 获取默认保存路径和文件名
        default_dir = os.path.dirname(self.parent().openpath) if hasattr(self.parent(), 'openpath') else ""
        default_filename = "processed_image.tif"
        default_path = os.path.join(default_dir, default_filename)

        # 使用文件对话框让用户选择保存路径和文件名
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存处理后的图像", default_path, "TIFF文件 (*.tif *.tiff);;所有文件 (*)"
        )
        if file_path:
            self.lineEdit_2.setText(file_path)


    def run_gray_process(self):
        """执行灰度拉伸（pushButton2事件）"""
        # 获取主窗口对象
        main_window = self.parent()
        if not main_window or not hasattr(main_window, "openpath"):
            QMessageBox.critical(self, "错误", "无法获取主窗口上下文")
            return

        # 获取输入参数
        try:
            truncated_value = float(self.lineEdit.text())
            if truncated_value < 0 or truncated_value > 100:
                raise ValueError("截断百分比应在0-100之间")
        except ValueError:
            QMessageBox.critical(self, "参数错误", "请输入有效的截断百分比（建议0.5）")
            return

        save_path = self.lineEdit_2.text().strip()
        if not save_path:
            QMessageBox.critical(self, "路径错误", "请选择保存文件名")
            return

        # 获取原始图像路径
        image_path = main_window.openpath
        if not image_path or not os.path.exists(image_path):
            QMessageBox.critical(self, "图像错误", "未找到有效的图像文件")
            return

        # 执行灰度拉伸
        try:
            # 使用img_utils读取栅格数据
            ds = read_dataset(image_path, parent=self)
            bands = read_band_data(ds)

            # 处理第一个波段（可根据需要修改波段索引）
            band_data = bands[1]

            # 调用灰度拉伸函数
            processed_img = gray_process(
                band_data,
                truncated_value=truncated_value,
                maxout=255,
                minout=0
            )

            # 使用img_utils保存栅格数据
            arr2raster(processed_img, save_path, image_path)

            # 显示成功信息
            QMessageBox.information(
                self, "成功",
                f"灰度拉伸完成\n保存路径：{save_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "处理失败", f"拉伸过程出错：{str(e)}")


# 图像样本显示窗口
class SampShowWindow(QMainWindow, Ui_SampleShowWin):
    def __init__(self, parent=None):
        super(SampShowWindow, self).__init__(parent)
        self.setupUi(self)

        # 设置GroupBox固定大小
        self.groupimages.setFixedSize(650, 650)

        # 创建网格布局
        self.grid_layout = QGridLayout()
        self.grid_layout.setHorizontalSpacing(50)
        self.grid_layout.setVerticalSpacing(50)
        self.grid_layout.setContentsMargins(50, 50, 50, 50)

        # 初始化图片标签列表
        self.image_labels = []
        for _ in range(9):
            label = QLabel()
            self.image_labels.append(label)
            self.grid_layout.addWidget(label, _ // 3, _ % 3)

        # 设置布局到GroupBox
        self.groupimages.setLayout(self.grid_layout)

        # 添加"未选择类别"提示
        self.placeholder_label = QLabel("未选择类别", self.groupimages)
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("font-size: 24px; color: gray;")
        self.placeholder_label.setGeometry(0, 0, self.groupimages.width(), self.groupimages.height())
        self.placeholder_label.show()

        # 获取images文件夹下的所有子文件夹
        self.image_folder = "Sample Set"

        # 创建映射字典：按钮英文名 -> (中文文件夹名, 颜色数组)
        self.mapping = {
            "curvedune": ("曲线型沙丘", [31, 119, 180]),
            "slopestreak": ("斜坡条纹", [44, 160, 44]),
            "texture": ("纹理", [127, 127, 127]),
            "rough": ("粗糙", [140, 86, 74]),
            "mixture": ("混合", [148, 103, 189]),
            "channel": ("沟槽", [152, 223, 138]),
            "straightdune": ("直线型沙丘", [174, 199, 232]),
            "mound": ("土丘", [196, 156, 148]),
            "ridge": ("山脊", [197, 176, 213]),
            "gully": ("冲沟", [214, 39, 40]),
            "craters": ("陨石坑群", [227, 119, 194]),
            "smooth": ("光滑", [247, 182, 210]),
            "cliff": ("悬崖", [255, 127, 14]),
            "mass": ("滑坡", [255, 152, 150]),
            "crater": ("陨石坑", [255, 187, 120])
        }

        # 创建按钮列表
        self.category_buttons = [
            self.btncurvedune, self.btnslopestreak, self.btntexture,
            self.btnrough, self.btnmixture, self.btnchannel,
            self.btnstraightdune, self.btnmound, self.btnridge,
            self.btngully, self.btncraters, self.btnsmooth,
            self.btncliff, self.btnmass, self.btncrater
        ]

        # 添加label列表
        self.color_labels = []

        # 按钮绑定对应名称文件夹
        for button in self.category_buttons:
            key = button.objectName()[3:]

            if key in self.mapping:
                chinese_name, color_array = self.mapping[key]
                folder_path = os.path.join(self.image_folder, chinese_name)

                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    # 创建对应的label
                    color_label = QLabel(button.parent())
                    color_label.setObjectName(f"lbl{key}")
                    color_label.setFixedSize(61, 21)

                    # 设置颜色（将RGB数组转换为十六进制颜色代码）
                    hex_color = f"#{color_array[0]:02x}{color_array[1]:02x}{color_array[2]:02x}"
                    color_label.setStyleSheet(f"background-color: {hex_color}; border: 1px solid black;")

                    # 将label放置在按钮右侧间隔40像素的位置
                    button_x = button.x()
                    button_width = button.width()
                    color_label.move(button_x + button_width + 40, button.y())
                    color_label.show()

                    self.color_labels.append(color_label)

                    button.clicked.connect(
                        lambda checked, c=chinese_name, col=color_array: self.load_category(c, col)
                    )
                    button.setText(chinese_name)
                    button.show()
                else:
                    button.hide()
            else:
                button.hide()

        # 设置lblpages文本居中显示
        self.lblpages.setAlignment(Qt.AlignCenter)  # 水平和垂直都居中
        self.lblpages.setStyleSheet("font-size: 24px;")

        # 设置lbltotal文本显示
        self.lbltotal.setStyleSheet("font-size: 24px;")

        # 初始化图片变量
        self.current_category = None
        self.image_files = []
        self.total_pages = 0
        self.current_page = 0

        # 连接翻页按钮信号
        self.imgup.clicked.connect(self.prev_page)
        self.imgdown.clicked.connect(self.next_page)
        self.btnclear.clicked.connect(self.clear_display)

    def get_categories(self):
        """获取images文件夹下的所有子文件夹"""
        categories = []
        if os.path.exists(self.image_folder):
            for item in os.listdir(self.image_folder):
                item_path = os.path.join(self.image_folder, item)
                if os.path.isdir(item_path):
                    categories.append(item)
        return sorted(categories)

    def load_category(self, chinese_name, color_array):
        """加载指定类别的图片"""
        self.current_category = chinese_name
        self.current_color = color_array
        folder_path = os.path.join(self.image_folder, chinese_name)

        # 获取该类别下的所有图片
        self.image_files = sorted(glob.glob(os.path.join(folder_path, "*.png")) +
                                  glob.glob(os.path.join(folder_path, "*.jpg")))
        self.total_pages = (len(self.image_files) + 8) // 9
        self.current_page = 0

        # 更新样本总量显示
        self.lbltotal.setText(f"样本总量:{len(self.image_files)}")

        # 隐藏"未选择类别"提示
        self.placeholder_label.hide()

        # 加载第一页
        self.load_page()

    def load_page(self):
        """加载当前页的图片"""
        # 更新页码显示
        self.lblpages.setText(f"{self.current_page + 1}/{self.total_pages}")

        if not self.image_files:
            # 如果没有图片，显示提示
            for i in range(9):
                if i == 4:
                    self.image_labels[i].setText("该类别无图片")
                    self.image_labels[i].setStyleSheet("font-size: 18px; color: gray;")
                    self.image_labels[i].setAlignment(Qt.AlignCenter)
                else:
                    self.image_labels[i].clear()
            return

        start_index = self.current_page * 9
        end_index = min(start_index + 9, len(self.image_files))

        # 加载当前页的图片
        for i in range(9):
            if i < (end_index - start_index):
                pixmap = QPixmap(self.image_files[start_index + i])
                pixmap = pixmap.scaled(150, 150)
                self.image_labels[i].setPixmap(pixmap)
            else:
                self.image_labels[i].clear()

    def prev_page(self):
        """上一页"""
        if self.current_page > 0:
            self.current_page -= 1
            self.load_page()

    def next_page(self):
        """下一页"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.load_page()

    def clear_display(self):
        """清空图片显示和统计信息"""
        # 重置页码显示
        self.lblpages.setText("0/0")

        # 重置样本总量显示
        self.lbltotal.setText("样本总量:0")

        # 清空图片显示
        for label in self.image_labels:
            label.clear()

        # 显示"未选择类别"提示
        self.placeholder_label.show()

        # 重置内部状态变量
        self.current_category = None
        self.image_files = []
        self.total_pages = 0
        self.current_page = 0


# 帮助显示窗口
class HelpWindow(QtWidgets.QDialog, Ui_HelpWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 把 UI 套到这个 QWidget/QDialog 实例上
        self.setupUi(self)
