# img_utils.py
from __future__ import annotations


import numpy as np
from osgeo import gdal, osr
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox, QLineEdit
from PyQt5.QtCore import Qt,pyqtSignal, QObject
import threading
from typing import Callable, Optional



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


def make_pixmap_from_band(arr, width, height):
    """把单波段 numpy 数组归一化、着色并转换成 QPixmap。"""
    mn, mx = float(arr.min()), float(arr.max())
    stretched = np.clip((arr - mn) / (mx - mn) * 255, 0, 255).astype(np.uint8)
    stretched = np.ascontiguousarray(stretched)
    h, w = stretched.shape
    qimg = QImage(stretched.data, w, h, w, QImage.Format_Grayscale8)
    pix = QPixmap.fromImage(qimg).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
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


