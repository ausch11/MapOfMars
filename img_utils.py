# img_utils.py
from __future__ import annotations


import numpy as np
from osgeo import gdal, osr
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox, QLineEdit
from PyQt5.QtCore import Qt,pyqtSignal, QObject
import threading
from typing import Callable, Optional


# 对数据进行拉伸处理
def gray_process(gray, truncated_value=0.5, maxout=255, minout=0):
    truncated_down = np.percentile(gray, truncated_value)
    truncated_up = np.percentile(gray, 100 - truncated_value)
    gray_new = ((maxout - minout) / (truncated_up - truncated_down)) * (gray - truncated_down)
    gray_new[gray_new < minout] = minout
    gray_new[gray_new > maxout] = maxout
    return np.uint8(gray_new)

def arr2raster(arr, raster_file, path):
    example = gdal.Open(path)
    prj = example.GetProjection()
    trans = example.GetGeoTransform()

    # 处理单波段/多波段数组维度
    if arr.ndim == 2:
        # 单波段：添加通道维度变为 (H, W, 1)
        num_bands = 1
        arr = arr[..., np.newaxis]  # 转换为3D数组以便统一处理
    elif arr.ndim == 3:
        # 多波段：直接使用第三维度作为波段数
        num_bands = arr.shape[2]
    else:
        raise ValueError(f"不支持非单波段或多波段数组")

    driver = gdal.GetDriverByName('GTiff')
    # 使用动态确定的波段数num_bands，而非固定取arr.shape[2]
    dst_ds = driver.Create(raster_file, arr.shape[1], arr.shape[0], num_bands, gdal.GDT_Byte)

    dst_ds.SetProjection(prj)
    dst_ds.SetGeoTransform(trans)

    # 将数组的各通道写入图片（统一处理单/多波段）
    for b in range(num_bands):
        dst_ds.GetRasterBand(b + 1).WriteArray(arr[:, :, b])

    dst_ds.FlushCache()

def read_dataset(path, parent=None):
    """打开 GDAL 数据集并做初步校验。返回 dataset 或抛出异常。"""
    ds = gdal.Open(path)
    if ds is None:
        msg = f"无法打开文件：\n{path}"
        if parent:
            QMessageBox.critical(parent, "错误", msg)
        raise IOError(msg)
    return ds


def extract_metadata(dataset):
    """从 GDAL dataset 中提取自身信息和投影信息，返回两个 dict。"""
    # 仿射变换
    gt = dataset.GetGeoTransform()
    xres, yres = gt[1], abs(gt[5])
    band0 = dataset.GetRasterBand(1)

    intrinsic = {
        "尺寸(像素)": f"{dataset.RasterXSize}×{dataset.RasterYSize}",
        "波段数": dataset.RasterCount,
        "数据类型": gdal.GetDataTypeName(band0.DataType),
        "分辨率": f"{xres:.3f}, {yres:.3f} m",
        "左上角坐标": f"{gt[0]:.3f}, {gt[3]:.3f}",
        "NoData 值": band0.GetNoDataValue() or "无",
        "驱动/格式": dataset.GetDriver().ShortName,
        "存储组织": dataset.GetMetadataItem("INTERLEAVE", "IMAGE_STRUCTURE") or "Unknown"
    }

    # 投影
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dataset.GetProjection())
    proj = {
        "投影类型": srs.GetAttrValue("PROJECTION", 0),
        "标准纬线": f"{srs.GetNormProjParm(osr.SRS_PP_STANDARD_PARALLEL_1, 0.0)}°",
        "中央经线": f"{srs.GetNormProjParm(osr.SRS_PP_CENTRAL_MERIDIAN, 0.0)}°",
        "假东距": f"{srs.GetNormProjParm(osr.SRS_PP_FALSE_EASTING, 0.0)} m",
        "假北距": f"{srs.GetNormProjParm(osr.SRS_PP_FALSE_NORTHING, 0.0)} m",
        "线性单位": srs.GetLinearUnitsName(),
        "椭球长半轴": f"{srs.GetSemiMajor():.1f} m"
    }

    return intrinsic, proj


def read_band_data(dataset):
    """读取所有波段到一个 dict: {band_index: numpy数组}"""
    bands = {}
    for i in range(1, dataset.RasterCount + 1):
        arr = dataset.GetRasterBand(i).ReadAsArray()
        if arr is None:
            raise ValueError(f"无法读取波段 {i}")
        bands[i] = arr
    return bands


def make_pixmap_from_band(arr, width=None, height=None):
    """把单波段 numpy 数组归一化、着色并转换成 QPixmap。"""
    mn, mx = float(arr.min()), float(arr.max())
    stretched = np.clip((arr - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)
    stretched = np.ascontiguousarray(stretched)
    # h, w = stretched.shape
    # qimg = QImage(stretched.data, w, h, w, QImage.Format_Grayscale8)
    # pix = QPixmap.fromImage(qimg).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    h, w = stretched.shape
    bytes_per_line = w  # 灰度每行字节数
    qimg = QImage(stretched.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()
    # .copy() 确保 QImage 拥有自己的数据副本，避免 numpy 内存生命周期问题

    pix = QPixmap.fromImage(qimg)  # 原始分辨率 pixmap
    return pix


def load_image_all(path, parent=None):
    """
    一步到位：读数据集、提取元数据、读波段数据
    返回 (dataset, intrinsic_dict, proj_dict, band_data_dict)
    """
    ds = read_dataset(path, parent)
    intrinsic, proj = extract_metadata(ds)
    bands = read_band_data(ds)
    return ds, intrinsic, proj, bands


def get_unique_colors(rgb):
    """
    通过读取的png，获得其中颜色索引，并与不同地貌类别对应
    输入：
      rgb: np.ndarray, shape = (H, W, 3), dtype = uint8
    返回：
      一个 list，元素是 (r, g, b) 三元组，表示图像中出现过的所有颜色
    """
    # 分离通道并提升到 uint32，避免溢出
    r = rgb[:, :, 0].astype(np.uint32)
    g = rgb[:, :, 1].astype(np.uint32)
    b = rgb[:, :, 2].astype(np.uint32)

    # 把 (R,G,B) 打包成一个 uint32 : 0xRRGGBB
    packed = (r << 16) | (g << 8) | b

    # 扁平化并去重
    uniq_vals = np.unique(packed.reshape(-1))

    # 拆回三元组
    colors = [
        ((val >> 16) & 0xFF,
         (val >> 8) & 0xFF,
         val & 0xFF)
        for val in uniq_vals
    ]
    return colors

def get_numeric_value(line_edit: QLineEdit,
                      *,
                      type_: type = int,
                      empty_msg: str = "请输入一个数字",
                      error_msg: str = "请输入一个合法的数字") -> int | float | None:
    """
    从给定的 QLineEdit 中读取文本并转换为数字。
    参数：
        line_edit: 要读取的 QLineEdit 对象
        type_:     要转换的类型，int 或 float（默认为 int）
        empty_msg: 如果文本为空时弹出的提示
        error_msg: 如果转换失败时弹出的提示
    返回值：
        成功时返回转换后的数字（int 或 float），失败时返回 None。
    """
    text = line_edit.text().strip()
    if not text:
        QMessageBox.warning(line_edit, "提示", empty_msg)
        return None
    try:
        value = type_(text)
    except ValueError:
        QMessageBox.critical(line_edit, "错误", error_msg)
        return None
    return value


def generate_color_legend(unique_colors, color_names,color_width=10, color_height=5,
                          font_size=12, padding_left=4, border_color="#ddd"):
    """
    生成颜色图例的HTML代码

    参数:
    - unique_colors: 颜色列表，每个颜色为(R, G, B)元组
    - color_names: 颜色名称字典，键为(R, G, B)元组，值为颜色名称
    - color_width: 色块宽度(像素)
    - color_height: 色块高度(像素)
    - font_size: 字体大小(像素)
    - padding_left: 文本左侧内边距(像素)
    - border_color: 边框颜色

    返回:
    - HTML字符串
    """
    html = f"""
    <table cellspacing="0" cellpadding="7" style="font-family:Arial; font-size:9.5pt;">
    """

    for color in unique_colors:
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        color_tuple = (r, g, b)
        color_name = color_names.get(color_tuple, "未命名类别")
        hexcol = f'#{r:02X}{g:02X}{b:02X}'

        html += f"""
        <tr>
          <td style="background-color:{hexcol}; width:{color_width}px; height:{color_height}px; 
                   border:1px solid {border_color};"></td>
          <td style="padding-left:{padding_left}px; font-size:{font_size}px; font-weight:450;">{color_name}</td>
        </tr>
        """

    html += "</table>"
    return html

# 进度条事件的设计
class ProgressManager:
    """
    进度条更新
    """
    def __init__(self, callback: Optional[Callable[[int], None]]):
        self.callback = callback if callback is not None else (lambda p: None)
        self.lock = threading.Lock()
        self.completed_fraction = 0.0  # fraction [0,1] already completed from previous subtasks
        self.current_sub_weight = 0.0
        self.current_sub_progress = 0.0
        self._last_reported = -1

    def start_subtask(self, weight: float):
        """weight: fraction of total (0..1)."""
        with self.lock:
            self.current_sub_weight = float(weight)
            self.current_sub_progress = 0.0
            self._report_locked()

    def update_subtask(self, sub_frac: float):
        """sub_frac: 0..1 progress within current subtask."""
        if sub_frac < 0: sub_frac = 0
        if sub_frac > 1: sub_frac = 1
        with self.lock:
            self.current_sub_progress = sub_frac
            self._report_locked()

    def finish_subtask(self):
        with self.lock:
            # mark subtask done and fold into completed_fraction
            self.completed_fraction += self.current_sub_weight
            # clamp numerical issues
            if self.completed_fraction > 1.0:
                self.completed_fraction = 1.0
            self.current_sub_weight = 0.0
            self.current_sub_progress = 0.0
            self._report_locked()

    def _report_locked(self):
        overall_frac = self.completed_fraction + self.current_sub_weight * self.current_sub_progress
        overall_pct = int(round(overall_frac * 100))
        if overall_pct != self._last_reported:
            try:
                self.callback(overall_pct)
            except Exception:
                pass
            self._last_reported = overall_pct

    def complete(self):
        """force to 100%"""
        with self.lock:
            self.completed_fraction = 1.0
            self.current_sub_weight = 0.0
            self.current_sub_progress = 1.0
            self._report_locked()


class ProgressSignals(QObject):
    progress_updated = pyqtSignal(int)  # 进度更新信号 (0-100)
    result_ready = pyqtSignal(np.ndarray)  # 分类结果信号
    error_occurred = pyqtSignal(str)  # 错误信号
    task_completed = pyqtSignal()  # 任务完成信号


