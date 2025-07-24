# img_utils.py
import os
import numpy as np
from osgeo import gdal, osr
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

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
        "中央经线": f"{srs.GetNormProjParm(osr.SRS_PP_CENTRAL_MERIDIAN,   0.0)}°",
        "假东距": f"{srs.GetNormProjParm(osr.SRS_PP_FALSE_EASTING,      0.0)} m",
        "假北距": f"{srs.GetNormProjParm(osr.SRS_PP_FALSE_NORTHING,     0.0)} m",
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
