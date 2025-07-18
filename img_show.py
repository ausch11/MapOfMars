import os
import numpy as np
from osgeo import gdal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from MapOfMars import process_utils
from process_utils import *

# 将错误信息以你设置的形式输出，而非在控制台中输出
gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')


class ImageProcessor:
    @staticmethod
    def load_image(image_path, statusbar, parent=None):
        """加载并处理遥感影像"""
        try:
            # 打开GDAL数据集
            dataset = gdal.Open(image_path)
            if dataset is None:
                statusbar.showMessage(f"无法打开文件: {image_path}")
                if parent:
                    QMessageBox.critical(parent, "错误", f"无法打开文件：\n{image_path}")
                return None, None, "无法打开文件"

            # 获取影像信息
            gt = dataset.GetGeoTransform() # 投影信息
            proj = dataset.GetProjection() # 1. 相关坐标信息
            proj_str = img_utils.parse_wkt_info(proj)
            semi_major = osr.SpatialReference()
            semi_major.ImportFromWkt(proj)
            radius = semi_major.GetSemiMajor() # 2.椭圆半径
            xres, yres = gt[1], abs(gt[5]) # 3.分辨率
            driver = dataset.GetDriver().ShortName # 4.保存形式
            interleave = dataset.GetMetadataItem("INTERLEAVE", "IMAGE_STRUCTURE") or "Unknown" # 5central_meridian.存储形式
            num_bands = dataset.RasterCount # 6. 波段数
            # 制作textedit中显示的信息
            base_info = (
                f"投影类型：{proj_str['projection']}\n"
                f"标准纬线：{proj_str['standard_parallel']}°\n"
                f"中央经线：{proj_str['central_meridian']}°\n"
                f"单位：{proj_str['unit']}\n"
                f"椭球体半径：{radius:.1f}m\n\n"
                f"分辨率：{xres:.3f}, {yres:.3f}\n"
                f"格式：{driver}\n"
                f"存储形式：{interleave}\n"
                f"波段数：{num_bands}"
            )
            # 显示在下方
            statusbar.showMessage(
                f"已加载: {os.path.basename(image_path)} | "
                f"波段数: {num_bands} | "
                f"尺寸: {dataset.RasterXSize}x{dataset.RasterYSize}"
            )

            # 读取波段数据
            band_data = {}
            for band_idx in range(1, num_bands + 1):
                band = dataset.GetRasterBand(band_idx)
                data = band.ReadAsArray()
                if data is None:
                    raise ValueError(f"无法读取波段 {band_idx}")
                band_data[band_idx] = data

            return dataset, band_data, base_info, None

        except Exception as e:
            statusbar.showMessage(f"错误: {str(e)}")
            if parent:
                QMessageBox.critical(parent, "加载错误", f"加载遥感影像失败：\n{e}")
            return None, None, None, str(e)

    @staticmethod
    def update_image(band_data, current_band, label, statusbar):
        """更新显示的图像"""
        if not band_data or current_band not in band_data:
            return None

        arr = band_data[current_band]
        min_val, max_val = arr.min(), arr.max()
        stretched = np.clip((arr - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
        stretched = np.ascontiguousarray(stretched)
        label._img_buffer = stretched  # 防止释放

        h, w = stretched.shape
        bytes_per_line = w
        qimg = QImage(stretched.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        statusbar.showMessage(f"波段: {current_band} | 数据: {min_val:.2f}-{max_val:.2f}")

        return pix
