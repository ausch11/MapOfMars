# coding: utf-8
# The code is written by Linghui


from skimage.segmentation import slic
from torchvision import transforms
from skimage import io
from skimage.measure import regionprops
import torch
import traceback
from torch.nn import functional as F
import matplotlib as mpl
from img_utils import *
import img_utils
from PyQt5.QtCore import QThread, pyqtSignal
from configs.C_model_1Dand2DCNN import densenet161
import math
import numpy as np
from numba import jit
import os
import cv2
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix
from PIL import Image

def image_patch(img2, slide_window, h, w):
    image = img2
    window_size = slide_window
    patch = np.zeros((slide_window, slide_window, h, w), dtype=np.uint8)

    for i in range(patch.shape[2]):
        for j in range(patch.shape[3]):
            patch[:, :, i, j] = img2[i: i + slide_window, j: j + slide_window]

    return patch

def superpixel_segmentation(img, superpixelsize, M, sigmaset, savepath, openpath):
    """
    输出超像素分割/标序的结果（4张图）
    """
    h, w = img.shape
    K = round((h * w) / (superpixelsize * superpixelsize))
    segments = slic(img, n_segments=K, compactness=M, sigma=sigmaset, channel_axis=None)

    # 画边界预览（float 0..1）
    out = mark_boundaries(img, segments)# 画出边界的slic分割图
    os.makedirs(savepath, exist_ok=True)
    plt.imsave(os.path.join(savepath, '超像素分割预览图.png'), out)
    arr2raster((out * 255).astype(np.uint8), os.path.join(savepath, '超像素分割地图.png'), openpath)

    # 画标序（编号）预览（把编号和红点绘到边界图上）
    labelled = out.copy()
    regions = regionprops(segments)
    # draw on labelled (ensure colours visible)
    for idx, region in enumerate(regions):
        r, c = region.centroid
        rr, cc = int(r), int(c)
        cv2.circle(labelled, (cc, rr), 3, color=(1, 0, 0), thickness=-1)
        cv2.putText(labelled, str(region.label), (cc + 5, rr + 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(1, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    plt.imsave(os.path.join(savepath, '超像素分割标序预览图.png'), labelled)
    arr2raster((labelled * 255).astype(np.uint8), os.path.join(savepath, '超像素分割标序地图.png'), openpath)

    return segments, regions, out, labelled

# 计算glcm特征
@jit(nopython=True)
def calcu_glcm_mean(glcm, nbit):
    mean = 0.0
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2
    return mean


@jit(nopython=True)
def calcu_glcm_variance(glcm, nbit=64):
    mean = 0.0
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2

    variance = 0.0
    for i in range(nbit):
        for j in range(nbit):
            variance += glcm[i, j] * (i - mean) ** 2

    return variance



@jit(nopython=True)
def calcu_glcm_homogeneity(glcm, nbit):
    homogeneity = 0.0
    for i in range(nbit):
        for j in range(nbit):
            homogeneity += glcm[i, j] / (1. + (i - j) ** 2)
    return homogeneity


@jit(nopython=True)
def calcu_glcm_contrast(glcm, nbit):
    contrast = 0.0
    for i in range(nbit):
        for j in range(nbit):
            contrast += glcm[i, j] * (i - j) ** 2
    return contrast


@jit(nopython=True)
def calcu_glcm_dissimilarity(glcm, nbit):
    dissimilarity = 0.0
    for i in range(nbit):
        for j in range(nbit):
            dissimilarity += glcm[i, j] * abs(i - j)
    return dissimilarity


@jit(nopython=True)
def calcu_glcm_entropy(glcm, nbit):
    eps = 0.00001
    entropy = 0.0
    for i in range(nbit):
        for j in range(nbit):
            if glcm[i, j] > 0:
                entropy -= glcm[i, j] * np.log10(glcm[i, j] + eps)
    return entropy


@jit(nopython=True)
def calcu_glcm_energy(glcm, nbit):
    energy = 0.0
    for i in range(nbit):
        for j in range(nbit):
            energy += glcm[i, j] ** 2
    return energy


@jit(nopython=True)
def calcu_glcm_correlation(glcm, nbit):
    mean = 0.0
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2
    variance = 0.0
    for i in range(nbit):
        for j in range(nbit):
            variance += glcm[i, j] * (i - mean) ** 2
    correlation = 0.0
    for i in range(nbit):
        for j in range(nbit):
            if variance == 0:
                return 0
            correlation += ((i - mean) * (j - mean) * (glcm[i, j])) / variance
    return correlation


@jit(nopython=True)
def calcu_glcm_Auto_correlation(glcm, nbit):
    auto_correlation = 0.0
    for i in range(nbit):
        for j in range(nbit):
            auto_correlation += glcm[i, j] * i * j
    return auto_correlation


def calcu_glcm(img, vmin=0, vmax=255, nbit=64, step=[2], angle=[0]):
    mi, ma = vmin, vmax
    bins = np.linspace(mi, ma + 1, nbit + 1)
    img1 = np.digitize(img, bins) - 1
    glcm = graycomatrix(img1, step, angle, levels=nbit)

    return glcm

def glcm(image, mi, ma, nbit, step, angle):
    """
    对图像计算所有glcm特征
    """
    img = image
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val == min_val:
        return np.zeros(144)  # 存在未空的图像，返回144维零向量
    img = np.uint8(255.0 * (img - min_val) / (max_val - min_val))  # normalization
    # Calcu GLCM
    glcm = calcu_glcm(img, mi, ma, nbit, step, angle)
    # Calcu Feature
    #
    mean_temp = []
    variance_temp = []
    homogeneity_temp = []
    contrast_temp = []
    dissimilarity_temp = []
    entropy_temp = []
    ASM_temp = []
    correlation_temp = []
    Auto_correlation_temp = []
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            glcm_cut = glcm[:, :, i, j]

            # normalize each GLCM
            glcm_cut = glcm_cut.astype(np.float64)
            glcm_sums = np.apply_over_axes(np.sum, glcm_cut, axes=(0, 1))
            glcm_sums[glcm_sums == 0] = 1
            glcm_cut /= glcm_sums

            mean = calcu_glcm_mean(glcm_cut, nbit)
            variance = calcu_glcm_variance(glcm_cut, nbit)
            homogeneity = calcu_glcm_homogeneity(glcm_cut, nbit)
            contrast = calcu_glcm_contrast(glcm_cut, nbit)
            dissimilarity = calcu_glcm_dissimilarity(glcm_cut, nbit)
            entropy = calcu_glcm_entropy(glcm_cut, nbit)
            ASM = calcu_glcm_energy(glcm_cut, nbit)
            correlation = calcu_glcm_correlation(glcm_cut, nbit)
            Auto_correlation = calcu_glcm_Auto_correlation(glcm_cut, nbit)

            mean_temp.append(np.float32(mean))
            variance_temp.append(np.float32(variance))
            homogeneity_temp.append(np.float32(homogeneity))
            contrast_temp.append(np.float32(contrast))
            dissimilarity_temp.append(np.float32(dissimilarity))
            entropy_temp.append(np.float32(entropy))
            ASM_temp.append(np.float32(ASM))
            correlation_temp.append(np.float32(correlation))
            Auto_correlation_temp.append(np.float32(Auto_correlation))

    all_feature = mean_temp + variance_temp + homogeneity_temp + contrast_temp + dissimilarity_temp + entropy_temp + ASM_temp + correlation_temp + Auto_correlation_temp

    return all_feature


def generate_patches(img, labelled, regions, pad, rm, superpixelsize_r, savepath, ctx_stripe,
                     data_transform, glcmfeature_mean, glcmfeature_std,
                     progress_callback=None):
    os.makedirs(os.path.join(savepath, "切片数据集"), exist_ok=True)
    os.makedirs(os.path.join(savepath, "标序切片数据集"), exist_ok=True)

    img_pad = np.pad(img, ((pad, pad), (pad, pad)), 'symmetric')  # 填充后的图像
    N = len(regions)

    # 创建内存映射文件（磁盘数组），避免内存溢出
    test_data1_path = os.path.join(savepath, "test_data1.dat")
    test_data2_path = os.path.join(savepath, "test_data2.dat")

    try:
        # 使用np.memmap创建磁盘映射文件
        test_data1_mem = np.memmap(test_data1_path, dtype=np.float32, mode='w+', shape=(N, 3, 224, 224))
        test_data2_mem = np.memmap(test_data2_path, dtype=np.float32, mode='w+', shape=(N, 144))  # 根据实际GLCM特征长度调整

        # 进度控制
        total_units = max(1, 2 * N)  # 防止 N=0
        units_done = 0
        # 节流：最多回调 100 次
        report_every_A = max(1, N // 100)
        report_every_B = max(1, N // 100)

        def report():
            if progress_callback:
                pct = int(round(units_done / total_units * 100))
                progress_callback(pct)

        # 处理图像切片和 GLCM 特征提取
        for i, region in enumerate(regions):
            cx, cy = region.centroid
            cx, cy = int(cx), int(cy)
            cx_pad, cy_pad = cx + pad, cy + pad
            temp = img_pad[cx_pad - rm:cx_pad + rm + 2, cy_pad - rm:cy_pad + rm + 2]
            # 保存灰度切片
            im = Image.fromarray(temp).convert('L')
            im.save(os.path.join(savepath, '切片数据集', f'{ctx_stripe}_{i}.png'))

            # CNN 三通道输入
            temp1 = np.dstack([temp] * 3)
            temp1 = Image.fromarray(np.uint8(temp1))
            temp1 = data_transform(temp1)
            # 将数据写入内存映射数组
            test_data1_mem[i, :, :, :] = np.asarray(temp1, dtype=np.float32)

            # GLCM 特征并标准化
            temp2 = img_pad[cx_pad - superpixelsize_r:cx_pad + superpixelsize_r + 1,
                    cy_pad - superpixelsize_r:cy_pad + superpixelsize_r + 1]
            glcmfeature_temp = glcm(temp2, 0, 255, 32,
                                    [2, 4, 8, 16],
                                    [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
            glcmfeature_temp = np.asarray(glcmfeature_temp, dtype=np.float32)

            # 标准化（确保 mean/std 长度匹配）
            glcm_len = glcmfeature_temp.shape[0]
            assert glcmfeature_mean.shape[0] == glcm_len and glcmfeature_std.shape[0] == glcm_len, \
                f"GLCM mean/std length mismatch: {glcmfeature_mean.shape[0]} vs {glcm_len}"
            glcmfeature_temp = (glcmfeature_temp - glcmfeature_mean[:glcm_len]) / (glcmfeature_std[:glcm_len] + 1e-12)

            # 将 GLCM 特征写入内存映射数组
            test_data2_mem[i, :] = glcmfeature_temp

            # 进度回调（按批次减少 UI 负担）
            units_done += 1
            if (i % report_every_A == 0) or (i == N - 1):
                report()

        # 保存带编号的标序切片预览（在原边界图上画绿色方框）
        out_pad = np.pad(labelled, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')

        for i, region in enumerate(regions):
            cx, cy = region.centroid
            cx, cy = int(cx), int(cy)
            cx_pad, cy_pad = cx + pad, cy + pad
            temp = out_pad[cx_pad - rm:cx_pad + rm + 2, cy_pad - rm:cy_pad + rm + 2, :].copy()
            cv2.rectangle(temp, (rm - superpixelsize_r, rm - superpixelsize_r),
                          (rm + superpixelsize_r, rm + superpixelsize_r), color=(0, 1, 0), thickness=1)
            plt.imsave(os.path.join(savepath, '标序切片数据集', f'{ctx_stripe}_{i}.png'), temp)

            units_done += 1
            if (i % report_every_B == 0) or (i == N - 1):
                report()

        # 最终确保进度 100%
        if progress_callback:
            progress_callback(100)

        # 将内存映射数据读入内存数组
        test_data1 = np.array(test_data1_mem)
        test_data2 = np.array(test_data2_mem)

        return test_data1, test_data2

    finally:
        # 确保删除临时文件
        try:
            # 删除内存映射对象
            del test_data1_mem
            del test_data2_mem
        except:
            pass

        try:
            # 删除磁盘文件
            if os.path.exists(test_data1_path):
                os.remove(test_data1_path)
            if os.path.exists(test_data2_path):
                os.remove(test_data2_path)
        except Exception as e:
            print(f"删除临时文件时出错: {e}")


# 马尔可夫随机场后处理分割的图像
# 初始版本
# @jit(nopython=True)
# def mrf_kernel(mrf_old, mrf_gamma, neighborhood_size):
#     """
#     单次 MRF 迭代核（numba-friendly）。
#     输入:
#       mrf_old: H x W x C (float32) - 每像素每类的概率
#       mrf_gamma: float
#       neighborhood_size: int (radius)
#     返回:
#       mrf_new: H x W x C (float32)
#     """
#     H, W, C = mrf_old.shape
#     mrf_new = np.zeros((H, W, C), dtype=np.float32)
#
#     for r in range(H):
#         for c in range(W):
#             # 当前像素的概率向量
#             pixel_probs = mrf_old[r, c, :]  # length C
#
#             # 邻域边界
#             r0 = r - neighborhood_size
#             if r0 < 0:
#                 r0 = 0
#             r1 = r + neighborhood_size + 1
#             if r1 > H:
#                 r1 = H
#
#             c0 = c - neighborhood_size
#             if c0 < 0:
#                 c0 = 0
#             c1 = c + neighborhood_size + 1
#             if c1 > W:
#                 c1 = W
#
#             neighbor_cnt = 0
#             votes = np.zeros(C, dtype=np.float32)
#
#             # 计算邻居投票（用邻居的 argmax）
#             for nr in range(r0, r1):
#                 for nc in range(c0, c1):
#                     if nr == r and nc == c:
#                         continue
#                     # 手写 argmax（numba 兼容）
#                     maxv = mrf_old[nr, nc, 0]
#                     lab = 0
#                     for kk in range(1, C):
#                         val = mrf_old[nr, nc, kk]
#                         if val > maxv:
#                             maxv = val
#                             lab = kk
#                     votes[lab] += 1.0
#                     neighbor_cnt += 1
#
#             # 计算 gibs（每个类别的权重）
#             if neighbor_cnt == 0:
#                 gibs = np.ones(C, dtype=np.float32)
#             else:
#                 # gibs_k = exp(-gamma*(neighbor_cnt - votes_k))
#                 gibs = np.exp(-mrf_gamma * (neighbor_cnt - votes))
#
#             # 结合并归一化，保护分母
#             combined = np.empty(C, dtype=np.float32)
#             s = 0.0
#             for kk in range(C):
#                 combined[kk] = gibs[kk] * pixel_probs[kk]
#                 s += combined[kk]
#             if s <= 0.0:
#                 # 退化处理：归一化原始 pixel_probs（若也为 0，则均匀分布）
#                 s2 = 0.0
#                 for kk in range(C):
#                     s2 += pixel_probs[kk]
#                 if s2 <= 0.0:
#                     # 全为 0 -> 均匀分布
#                     for kk in range(C):
#                         mrf_new[r, c, kk] = 1.0 / C
#                 else:
#                     for kk in range(C):
#                         mrf_new[r, c, kk] = pixel_probs[kk] / s2
#             else:
#                 for kk in range(C):
#                     mrf_new[r, c, kk] = combined[kk] / s
#     return mrf_new

# 修改后的mrf_kernel，增加存储中间磁盘数据的部分
@jit(nopython=True)
def mrf_kernel(mrf_old, mrf_gamma, neighborhood_size, r0, r1, c0, c1):
    """
    numba-friendly 子窗口 MRF 计算（返回一个 tile）。
    mrf_old: full H x W x C array (可为 memmap 或 ndarray)，dtype float32
    mrf_gamma: float32 (or float) -> 内部转换为 float32
    neighborhood_size: int
    (r0,r1,c0,c1): 子窗口（像素索引，r0<=r<r1, c0<=c<c1）
    返回: tile array with shape (r1-r0, c1-c0, C) dtype float32
    """
    H, W, C = mrf_old.shape
    # 确保子窗口合法
    if r0 < 0:
        r0 = 0
    if c0 < 0:
        c0 = 0
    if r1 > H:
        r1 = H
    if c1 > W:
        c1 = W

    gamma = np.float32(mrf_gamma)

    out_h = r1 - r0
    out_w = c1 - c0
    tile = np.zeros((out_h, out_w, C), dtype=np.float32)

    # 为计算邻域，实际需要访问的扩展窗口范围
    # 但这里我们直接从 mrf_old 读取任意位置（上层传入的是 padded coords），
    # 所以在上层我们会把 pad 区间传入（即 r0,r1,c0,c1 已含必要 padding）。
    for rr in range(r0, r1):
        for cc in range(c0, c1):
            # current pixel probs
            pixel_probs = mrf_old[rr, cc, :]  # (C,)
            # neighbor bounds
            n_r0 = rr - neighborhood_size
            if n_r0 < 0:
                n_r0 = 0
            n_r1 = rr + neighborhood_size + 1
            if n_r1 > H:
                n_r1 = H
            n_c0 = cc - neighborhood_size
            if n_c0 < 0:
                n_c0 = 0
            n_c1 = cc + neighborhood_size + 1
            if n_c1 > W:
                n_c1 = W

            neighbor_cnt = 0
            votes = np.zeros(C, dtype=np.float32)

            # gather votes from neighbors (argmax of neighbor probs)
            for nr in range(n_r0, n_r1):
                for nc in range(n_c0, n_c1):
                    if nr == rr and nc == cc:
                        continue
                    maxv = mrf_old[nr, nc, 0]
                    lab = 0
                    for kk in range(1, C):
                        val = mrf_old[nr, nc, kk]
                        if val > maxv:
                            maxv = val
                            lab = kk
                    votes[lab] += np.float32(1.0)
                    neighbor_cnt += 1

            # compute gibs (float32)
            if neighbor_cnt == 0:
                gibs = np.ones(C, dtype=np.float32)
            else:
                gibs = np.empty(C, dtype=np.float32)
                ncnt_f = np.float32(neighbor_cnt)
                for kk in range(C):
                    # use math.exp per-element and cast to float32
                    gibs[kk] = np.float32(math.exp(-gamma * (ncnt_f - votes[kk])))

            # combine and normalize
            combined = np.empty(C, dtype=np.float32)
            s = np.float32(0.0)
            for kk in range(C):
                combined[kk] = gibs[kk] * pixel_probs[kk]
                s += combined[kk]

            if s <= np.float32(0.0):
                s2 = np.float32(0.0)
                for kk in range(C):
                    s2 += pixel_probs[kk]
                if s2 <= np.float32(0.0):
                    val = np.float32(1.0 / C)
                    for kk in range(C):
                        tile[rr - r0, cc - c0, kk] = val
                else:
                    for kk in range(C):
                        tile[rr - r0, cc - c0, kk] = pixel_probs[kk] / s2
            else:
                for kk in range(C):
                    tile[rr - r0, cc - c0, kk] = combined[kk] / s

    return tile



# 原始生成代码
# def generate_patches(img, segments, regions, pad, rm, superpixelsize_r, savepath, ctx_stripe,
#                      data_transform, glcmfeature_mean, glcmfeature_std,
#                      progress_callback=None):
#     """
#     获取图像切片，改变其格式以进行训练
#     progress_callback用于进度条的更新
#     """
#     os.makedirs(os.path.join(savepath, "切片数据集"), exist_ok=True)
#     os.makedirs(os.path.join(savepath, "标序切片数据集"), exist_ok=True)
#
#     img_pad = np.pad(img, ((pad, pad), (pad, pad)), 'symmetric')# 填充后的图像
#     N = len(regions)
#     test_data1 = []
#     test_data2 = []
#
#     # 进度控制
#     total_units = max(1, 2 * N)  # 防止 N=0
#     units_done = 0
#     # 节流：最多回调 100 次
#     report_every_A = max(1, N // 100)
#     report_every_B = max(1, N // 100)
#
#     def report():
#         if progress_callback:
#             pct = int(round(units_done / total_units * 100))
#             progress_callback(pct)
#
#     for i, region in enumerate(regions):
#         cx, cy = region.centroid
#         cx, cy = int(cx), int(cy)
#         cx_pad, cy_pad = cx + pad, cy + pad
#         temp = img_pad[cx_pad - rm:cx_pad + rm + 2, cy_pad - rm:cy_pad + rm + 2]
#         # 保存灰度切片
#         im = Image.fromarray(temp).convert('L')
#         im.save(os.path.join(savepath, '切片数据集', f'{ctx_stripe}_{i}.png'))
#
#         # CNN 三通道输入
#         temp1 = np.dstack([temp] * 3)
#         temp1 = Image.fromarray(np.uint8(temp1))
#         temp1 = data_transform(temp1)
#         test_data1.append(np.asarray(temp1, dtype=np.float32))
#         # GLCM 特征并标准化
#         temp2 = img_pad[cx_pad - superpixelsize_r:cx_pad + superpixelsize_r + 1,
#                         cy_pad - superpixelsize_r:cy_pad + superpixelsize_r + 1]
#         glcmfeature_temp = glcm(temp2, 0, 255, 32,
#                                 [2, 4, 8, 16],
#                                 [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
#         glcmfeature_temp = np.asarray(glcmfeature_temp, dtype=np.float32)
#
#         # 标准化（确保 mean/std 长度匹配）
#         glcm_len = glcmfeature_temp.shape[0]
#         assert glcmfeature_mean.shape[0] == glcm_len and glcmfeature_std.shape[0] == glcm_len, \
#             f"GLCM mean/std length mismatch: {glcmfeature_mean.shape[0]} vs {glcm_len}"
#         glcmfeature_temp = (glcmfeature_temp - glcmfeature_mean[:glcm_len]) / (glcmfeature_std[:glcm_len] + 1e-12)
#         test_data2.append(glcmfeature_temp)
#
#         # 进度回调（按批次减少 UI 负担）
#         units_done += 1
#         if (i % report_every_A == 0) or (i == N - 1):
#             report()
#
#         # for k in range(glcmfeature_temp.shape[0]):
#         #     glcmfeature_temp[k] = (glcmfeature_temp[k] - glcmfeature_mean[k]) / glcmfeature_std[k]
#         # test_data2.append(glcmfeature_temp)
#
#     # 保存带编号的标序切片预览（在原边界图上画绿色方框）
#     out = mark_boundaries(img, segments)
#     out_pad = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')
#
#     for i, region in enumerate(regions):
#         cx, cy = region.centroid
#         cx, cy = int(cx), int(cy)
#         cx_pad, cy_pad = cx + pad, cy + pad
#         temp = out_pad[cx_pad - rm:cx_pad + rm + 2, cy_pad - rm:cy_pad + rm + 2, :]
#         cv2.rectangle(temp, (rm - superpixelsize_r, rm - superpixelsize_r),
#                       (rm + superpixelsize_r, rm + superpixelsize_r), color=(0, 1, 0), thickness=1)
#         plt.imsave(os.path.join(savepath, '标序切片数据集', f'{ctx_stripe}_{i}.png'), temp)
#
#         units_done += 1
#         if (i % report_every_B == 0) or (i == N - 1):
#             report()
#
#     # 最终确保进度 100%
#     if progress_callback:
#         progress_callback(100)
#
#     return np.asarray(test_data1, dtype=np.float32), np.asarray(test_data2, dtype=np.float32)

from mmpretrain.apis import init_model
from mmpretrain.utils import register_all_modules

def _extract_logits_from_output(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        for k in ('logits', 'pred', 'out', 'outputs'):
            if k in out:
                return out[k]
        return list(out.values())[0]
    if isinstance(out, (list, tuple)):
        return out[0]
    raise RuntimeError(f"Unsupported model output type: {type(out)}")

def load_model(model_spec: dict, device: str = 'cuda'):
    """
    支持两种 model_spec 类型：
      - 'mmpretrain': {'type':'mmpretrain','config':..., 'ckpt':..., 'two_inputs':bool}
      - 'state_dict': {'type':'state_dict','constructor':callable, 'ckpt':..., 'strict':bool, 'two_inputs':bool}

    返回 (model, predict_fn, two_inputs)
    """

    model_type = model_spec.get('type')
    two_inputs = bool(model_spec.get('two_inputs', False))

    # 选择 device：优先使用用户指定，但如果 cuda 不可用则降到 cpu
    device_t = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    if model_type == 'mmpretrain':
        # 原来的 mmpretrain 加载逻辑
        cfg = model_spec.get('config')
        ckpt = model_spec.get('ckpt')
        if cfg is None or ckpt is None:
            raise ValueError("mmpretrain需要config和ckpt文件")
        register_all_modules()
        # init_model 可能会执行 config 中的 imports（确保 sys.path 已配置）
        model = init_model(cfg, ckpt, device=str(device_t))
        model.eval()

        def predict_fn(x):
            out = model(x)
            logits = _extract_logits_from_output(out)
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, device=device_t)
            return logits

        return model, predict_fn, two_inputs

    elif model_type == 'state_dict':
        # state_dict 型加载：需要 constructor 和 ckpt 路径
        constructor = model_spec.get('constructor')
        ckpt = model_spec.get('ckpt')
        if constructor is None or ckpt is None:
            raise ValueError("state_dict需要constructora和ckpt文件")

        # 构造模型 skeleton
        model = constructor()
        # 加载 checkpoint（先加载到 cpu，之后再 .to(device)）
        try:
            sd = torch.load(ckpt, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"加载 checkpoint 失败: {ckpt}\n原始错误: {e}")

        # 移动到目标设备并设为 eval 模式
        model.to(device_t)
        model.eval()
        model.load_state_dict(sd)

        def predict_fn(x):
            # 两路输入
            if isinstance(x, (list, tuple)):
                x0 = x[0].to(device_t)
                x1 = x[1].to(device_t)
                out = model(x0, x1)
            else:
                out = model(x.to(device_t))
            logits = _extract_logits_from_output(out)
            # 确保 logits 在正确设备
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, device=device_t)
            else:
                logits = logits.to(device_t)
            return logits

        return model, predict_fn, two_inputs

    else:
        raise ValueError(f"Unsupported model_spec type: {model_type}")

def predict_batch(predict_fn, x_batch, two_inputs=False, x2_batch=None, device='cuda', num_classes=None):
    """
    通用的 batch 前向函数（被 classify_patches 调用）。

    Args:
      predict_fn: load_model 返回的 predict_fn（callable）
      x_batch: torch.Tensor, shape (B, C, H, W) 或 numpy -> 已在调用处保证为 tensor
      two_inputs: bool, 是否为两路输入
      x2_batch: torch.Tensor for second input if two_inputs
      device: str or torch.device
      num_classes: int or None -> 若提供，则对 logits 做截断或右侧补零以匹配 num_classes

    Returns:
      logits (torch.Tensor on device), probs (numpy array on CPU, shape (B, num_classes or C_out))
    """
    # 确保 tensor 在正确 device
    device_t = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    if not isinstance(x_batch, torch.Tensor):
        x_batch = torch.as_tensor(x_batch)
    x_batch = x_batch.to(device_t)

    with torch.no_grad():
        if two_inputs:
            if not isinstance(x2_batch, torch.Tensor):
                x2_batch = torch.as_tensor(x2_batch)
            x2_batch = x2_batch.to(device_t)

            out = predict_fn((x_batch, x2_batch))
        else:
            out = predict_fn(x_batch)

        if not isinstance(out, torch.Tensor):
            logits = torch.as_tensor(out, device=device_t)
        else:
            logits = out.to(device_t)

        # 对 logits 的通道数做对齐（如果 num_classes 提供）
        if num_classes is not None:
            C_out = logits.shape[1]
            if C_out != num_classes:
                # 截断或补零
                if C_out > num_classes:
                    logits = logits[:, :num_classes]
                else:
                    pad = torch.zeros((logits.shape[0], num_classes - C_out), device=logits.device, dtype=logits.dtype)
                    logits = torch.cat([logits, pad], dim=1)

        probs = F.softmax(logits, dim=1).cpu().numpy()
    return logits, probs


def classify_patches(test_data1, test_data2, model_spec, num_classes, batch_size=64,
                     progress_callback=None, device='cuda'):
    """
    使用 predict_batch 统一前向和对齐逻辑。

    Args:
      test_data1: numpy array, shape (N, C, H, W)
      test_data2: numpy array, shape (N, F)
      model_spec: dict for load_model
      num_classes: expected number of classes (int)
      batch_size: int
      progress_callback: callable(progress_int)
      device: 'cuda' or 'cpu'

    Returns:
      predictions: (N,) int
      scores: (N, num_classes) float
    """
    # 校验输入
    print("执行分类")

    model, predict_fn, two_inputs = load_model(model_spec, device=device)

    # 构造 dataloader（使用 TensorDataset，但我们保持 numpy->tensor 转换在 predict_batch 中）
    dataset1 = torch.utils.data.TensorDataset(torch.from_numpy(test_data1))
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=False)

    if two_inputs:
        dataset2 = torch.utils.data.TensorDataset(torch.from_numpy(test_data2[:, np.newaxis, :]))
        loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False)
        if len(loader1) != len(loader2):
            raise RuntimeError(f"two_inputs 模式下 batch 数不一致: {len(loader1)} vs {len(loader2)}")
    else:
        loader2 = None

    preds_list = []
    scores_list = []

    total_batches = max(len(loader1), len(loader2)) if two_inputs else len(loader1)

    # 遍历 batches
    if two_inputs:
        for b_idx, ((x1,), (x2,)) in enumerate(zip(loader1, loader2)):
            # x1: tensor (B, C, H, W) ; x2: tensor (B, 1, F) -> 我们需要把 x2 squeeze 到 (B, F)
            logits, probs = predict_batch(predict_fn, x1, two_inputs=True, x2_batch=x2, device=device, num_classes=num_classes)
            preds = np.argmax(probs, axis=1)
            preds_list.append(preds)
            scores_list.append(probs)

            if progress_callback:
                pct = int(round((b_idx + 1) / float(total_batches) * 100))
                progress_callback(pct)
    else:
        for b_idx, (x_batch,) in enumerate(loader1):
            logits, probs = predict_batch(predict_fn, x_batch, two_inputs=False, x2_batch=None, device=device, num_classes=num_classes)
            preds = np.argmax(probs, axis=1)
            preds_list.append(preds)
            scores_list.append(probs)

            if progress_callback:
                pct = int(round((b_idx + 1) / float(total_batches) * 100))
                progress_callback(pct)

    # concat
    if len(preds_list) == 0:
        return np.array([], dtype=np.int32), np.zeros((0, num_classes), dtype=np.float32)

    predictions = np.concatenate(preds_list, axis=0)
    scores = np.concatenate(scores_list, axis=0)

    if progress_callback:
        progress_callback(100)

    return predictions, scores

# def classify_patches(test_data1, test_data2, model_path, num_classes, batch_size=64,
#                      progress_callback=None):
#     model = densenet161(num_classes=num_classes)
#     assert isinstance(model_path, object)
#     checkpoint = torch.load(model_path)
#     model.load_state_dict(checkpoint)
#     model.cuda()
#     model.eval()
#
#     dataset1 = torch.utils.data.TensorDataset(torch.from_numpy(test_data1))
#     dataset2 = torch.utils.data.TensorDataset(torch.from_numpy(test_data2[:, np.newaxis, :]))
#     loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=False)
#     loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False)
#
#     total_batches = max(len(loader1), len(loader2))
#     predictions = []
#     scores = []
#     with torch.no_grad():
#         for b_idx, ((x1,), (x2,)) in enumerate(zip(loader1, loader2)):
#             x1 = x1.cuda()
#             x2 = x2.cuda()
#             y_hat = model(x1, x2)
#             predictions.append(torch.argmax(y_hat, dim=1).cpu().numpy())
#             scores.append(F.softmax(y_hat, dim=1).cpu().numpy())
#
#             if progress_callback:
#                 pct = int(round((b_idx + 1) / float(total_batches) * 100))
#                 progress_callback(pct)
#
#     predictions = np.concatenate(predictions, axis=0)
#     scores = np.concatenate(scores, axis=0)
#
#     if progress_callback:
#         progress_callback(100)
#
#     return predictions, scores


# 增加中间存储磁盘数据，增加粒度回调
# def apply_mrf_and_save(segments, regions, scores, num_classes, savepath, openpath, cm,
#                        mrf_iterations=5, mrf_gamma=0.3, neighborhood_size=11,
#                        progress_callback=None, overlay_callback=None,
#                        base_tile=512, replace_retry=20, replace_wait=0.08):

def apply_mrf_and_save(segments, regions, scores, num_classes, savepath, openpath, cm,
                        mrf_iterations=5, mrf_gamma=0.3, neighborhood_size=11,
                        progress_callback=None, base_tile=512, replace_retry=20, replace_wait=0.08):
    """
    改进版 apply_mrf_and_save：
      - 使用 memmap 存储概率场
      - 按 tile 级别做细粒度进度回调（带时间节流和批量节流）
      - 提供重试替换与 Windows 上释放 memmap 的辅助
    """
    import os
    import time
    import gc
    import tempfile
    import numpy as np
    import matplotlib.pyplot as plt

    # ----------------- 辅助函数 -----------------
    def _close_memmap(mm):
        """尽力释放 memmap 的文件句柄（Windows 上必需）"""
        if mm is None:
            return
        try:
            mm.flush()
        except Exception:
            pass
        # 尝试关闭内部 _mmap（numpy 实现细节）
        try:
            m = getattr(mm, "_mmap", None)
            if m is not None:
                try:
                    m.close()
                except Exception:
                    pass
        except Exception:
            pass
        # 删除变量引用并强制回收
        try:
            del mm
        except Exception:
            pass
        gc.collect()
        # 给操作系统短暂时间回收文件句柄
        time.sleep(0.03)

    # ----------------- 基本准备 -----------------
    h, w = segments.shape
    dtype = np.float32
    est_bytes = int(h) * int(w) * int(num_classes) * 4

    temp_dir = tempfile.mkdtemp(dir=savepath)
    tmp_base = os.path.join(temp_dir, "mrf_prob_memmap.dat")

    # 创建初始 memmap 并填充 0
    mrf_old = np.memmap(tmp_base, dtype=dtype, mode='w+', shape=(h, w, num_classes))
    mrf_old[:] = 0.0

    # 填充区域概率（按像素）
    for j, region in enumerate(regions):
        coords = region.coords
        if coords is None or coords.size == 0:
            continue
        rows = coords[:, 0].astype(np.intp)
        cols = coords[:, 1].astype(np.intp)
        mrf_old[rows, cols, :] = scores[j, :].astype(dtype)
    mrf_old.flush()

    # 分块参数
    tile_h = base_tile if h > base_tile else h
    tile_w = base_tile if w > base_tile else w
    overlap = int(neighborhood_size)

    # 计算 tile grid 与计数（用于进度）
    row_starts = list(range(0, h, tile_h))
    col_starts = list(range(0, w, tile_w))
    num_row_blocks = len(row_starts)
    num_col_blocks = len(col_starts)
    num_tiles = num_row_blocks * num_col_blocks
    # 总单位：mrf_iterations * tiles + 分类最终读取按行块 + 少量收尾步骤
    final_class_blocks = num_row_blocks  # 最终按行读取块来计算 argmax
    total_units = max(1, mrf_iterations * num_tiles + final_class_blocks + 2)

    units_done = 0
    # 节流参数：时间和数量
    min_report_interval = 0.05  # 最小时间间隔（秒）
    report_every_tiles = max(1, num_tiles // 200)  # 至多约200次/迭代（可调整）
    last_report_time = 0.0

    def _maybe_report(force=False):
        nonlocal last_report_time, units_done
        if progress_callback is None:
            return
        now = time.time()
        if not force:
            # 时间节流与数量节流二者都满足时才上报
            if (now - last_report_time) < min_report_interval:
                return
        pct = int(round(units_done / float(total_units) * 100))
        pct = max(0, min(100, pct))
        try:
            progress_callback(pct)
        except Exception:
            pass
        last_report_time = now

    try:
        for it in range(mrf_iterations):
            tmp_iter = tmp_base + f".iter{it}"
            # 新 memmap
            mrf_new = np.memmap(tmp_iter, dtype=dtype, mode='w+', shape=(h, w, num_classes))
            mrf_new[:] = 0.0

            # 遍历每个 tile（包含 overlap region），调用 mrf_kernel
            tile_counter = 0
            for r_idx, r_start in enumerate(row_starts):
                r_end = min(r_start + tile_h, h)
                pr0 = max(0, r_start - overlap)
                pr1 = min(h, r_end + overlap)
                for c_idx, c_start in enumerate(col_starts):
                    c_end = min(c_start + tile_w, w)
                    pc0 = max(0, c_start - overlap)
                    pc1 = min(w, c_end + overlap)

                    # 获取 tile（mrf_kernel 从 mrf_old 读取需要的 region 并返回 tile）
                    tile = mrf_kernel(mrf_old, mrf_gamma, neighborhood_size, pr0, pr1, pc0, pc1)

                    # 计算中心区域索引，将 tile 中心写回 mrf_new
                    inner_r0 = r_start - pr0
                    inner_r1 = inner_r0 + (r_end - r_start)
                    inner_c0 = c_start - pc0
                    inner_c1 = inner_c0 + (c_end - c_start)
                    mrf_new[r_start:r_end, c_start:c_end, :] = tile[inner_r0:inner_r1, inner_c0:inner_c1, :]

                    tile_counter += 1
                    units_done += 1  # 一个 tile 完成算一个单位

                    # 节流上报：每经过一定数量 tile 或时间到达则上报
                    if (tile_counter % report_every_tiles == 0) or ((time.time() - last_report_time) >= min_report_interval):
                        _maybe_report()

                # 每完成一行块，flush 一次（控制 IO）
                try:
                    mrf_new.flush()
                except Exception:
                    pass

            # 完成本迭代所有 tiles
            # 尝试 flush & 释放 mrf_new/mrf_old，然后替换文件
            try:
                mrf_new.flush()
            except Exception:
                pass

            # 释放引用并尝试关闭 memmap（尽力）
            _close_memmap(mrf_new)
            _close_memmap(mrf_old)

            # 重试替换 tmp_iter -> tmp_base（使用 os.replace 做原子替换）
            replaced = False
            last_exc = None
            for attempt in range(replace_retry):
                try:
                    # 使用 os.replace（在同一文件系统上原子）
                    os.replace(tmp_iter, tmp_base)
                    replaced = True
                    break
                except Exception as e:
                    last_exc = e
                    time.sleep(replace_wait)

            if not replaced:
                # 抛出明确错误：替换失败
                raise RuntimeError(f"无法完成临时文件替换（{tmp_iter} -> {tmp_base}），最后错误: {last_exc!r}")

            # 重新打开 canonical memmap 作为 mrf_old（用于下一迭代或最终读取）
            mrf_old = np.memmap(tmp_base, dtype=dtype, mode='r+', shape=(h, w, num_classes))

            # 强制上报本次迭代结束（不论节流）
            _maybe_report(force=True)

        # ----------------- 迭代结束：按块计算最终类别（分块读取） -----------------
        mrf_classes = np.empty((h, w), dtype=np.int32)
        # 将 final_class_blocks 也纳入进度（每处理一行块上报）
        for r_idx, r_start in enumerate(row_starts):
            r_end = min(r_start + tile_h, h)
            # 读取整行块（r_start:r_end, :, :）
            block = np.array(mrf_old[r_start:r_end, :, :])
            block_cls = np.argmax(block, axis=2).astype(np.int32)
            mrf_classes[r_start:r_end, :] = block_cls

            units_done += 1
            _maybe_report()

    finally:
        # 尽力清理 memmap & 临时文件
        try:
            _close_memmap(mrf_old)
        except Exception:
            pass
        # 删除 temp_dir 下残留文件
        try:
            if os.path.exists(temp_dir):
                for fname in os.listdir(temp_dir):
                    fpath = os.path.join(temp_dir, fname)
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
                try:
                    os.rmdir(temp_dir)
                except Exception:
                    pass
        except Exception:
            pass

    # ----------------- 保存与 overlay（不计入太多进度但会最终上报） -----------------
    try:
        plt.imsave(os.path.join(savepath, 'MRF分类结果预览图.png'),
                   mrf_classes, cmap=cm, vmin=0, vmax=max(1, num_classes - 1))
    except Exception as e:
        print(f"保存预览图失败: {e}")

    try:
        # 你原来的 arr2raster 入口（可能依赖 skimage.io）
        arr2raster(io.imread(os.path.join(savepath, 'MRF分类结果预览图.png')),
                   os.path.join(savepath, 'MRF分类结果地图.png'), openpath)
    except Exception as e:
        print(f"arr2raster 失败: {e}")

    # 最终强制上报 100%
    if progress_callback:
        try:
            progress_callback(100)
        except Exception:
            pass

    return mrf_classes

# 原始对分类进行保存的代码
# def apply_mrf_and_save(segments, regions, scores, num_classes, savepath, openpath, cm,
#                        mrf_iterations=5, mrf_gamma=0.3, neighborhood_size=11,
#                        progress_callback=None, overlay_callback=None):
#     h, w = segments.shape
#     scores_map_3 = np.empty((h, w, num_classes), dtype=np.float32)
#     for j, region in enumerate(regions):
#         yy = (region.coords[:, 0], region.coords[:, 1])
#         scores_map_3[yy] = scores[j, :].astype(np.float32)
#
#     # mrf_probabilities = MRF(scores_map_3.astype(np.float32),
#     #                         mrf_iterations, mrf_gamma, neighborhood_size)
#     mrf_prob = scores_map_3.astype(np.float32, copy=False)
#     for it in range(mrf_iterations):
#         mrf_prob = mrf_kernel(mrf_prob, float(mrf_gamma), int(neighborhood_size))
#
#         if progress_callback:
#             pct = int(round((it + 1) / float(mrf_iterations) * 100))
#             progress_callback(pct)
#
#     mrf_classes = np.argmax(mrf_prob, axis=2)
#
#     # 保存MRF结果预览与地理图
#     plt.imsave(os.path.join(savepath, 'MRF分类结果预览图.png'), mrf_classes, cmap=cm, vmin=0, vmax=num_classes)
#     arr2raster(io.imread(os.path.join(savepath, 'MRF分类结果预览图.png')),
#                os.path.join(savepath, 'MRF分类结果地图.png'), openpath)
#
#     # 生成用于 overlay 的 RGB 预览（使用同样的 colormap，提取前三个通道）
#     cmap = mpl.cm.get_cmap(cm)
#     norm = mpl.colors.Normalize(vmin=0, vmax=max(1, num_classes - 1))
#     colored = cmap(norm(mrf_classes))  # RGBA float 0..1
#     prcresult_show = (colored[:, :, :3] * 255).astype(np.uint8)
#
#     # 如果有 overlay_callback，回传 RGB 图像
#     if overlay_callback:
#         overlay_callback(prcresult_show)
#
#     if progress_callback:
#         progress_callback(100)
#
#     return mrf_classes


def slic_map_pipeline(openpath, savepath, superpixelsize, window_size, M,
                      progress_callback=None, overlay_callback=None,
                      model_spec=None,
                      glcmfeature_mean_path='data/final_mean.npy',
                      glcmfeature_std_path='data/final_std.npy'):
    if progress_callback:
        progress_callback(0)

    # minimal ProgressManager-like behavior (if you have img_utils.ProgressManager you can plug it back)
    pm = img_utils.ProgressManager(progress_callback)

    weights = {
        'preproc_seg': 0.05,
        'generate_patches': 0.15,
        'classify': 0.25,
        'cls_patch': 0.15,
        'apply_mrf': 0.25,
        'mrf_patch': 0.15
    }
    sigmaset = 5
    if window_size == 200:
        rm = 99; pad = 100
    else:
        rm = 99; pad = 100
    superpixelsize_r = 60
    os.makedirs(savepath, exist_ok=True)

    custom_colors = [
        [31 / 255, 119 / 255, 180 / 255],
        [174 / 255, 199 / 255, 232 / 255],
        [255 / 255, 127 / 255, 14 / 255],
        [255 / 255, 187 / 255, 120 / 255],
        [44 / 255, 160 / 255, 44 / 255],
        [152 / 255, 223 / 255, 138 / 255],
        [214 / 255, 39 / 255, 40 / 255],
        [255 / 255, 152 / 255, 150 / 255],
        [148 / 255, 103 / 255, 189 / 255],
        [197 / 255, 176 / 255, 213 / 255],
        [140 / 255, 86 / 255, 75 / 255],
        [196 / 255, 156 / 255, 148 / 255],
        [227 / 255, 119 / 255, 194 / 255],
        [247 / 255, 182 / 255, 210 / 255],
        [127 / 255, 127 / 255, 127 / 255]
    ]
    hyper_params = {"num_classes":15}
    n = int(hyper_params["num_classes"])
    cm = mpl.colors.LinearSegmentedColormap.from_list("Custom", custom_colors, n)

    try:
        pm.start_subtask(weights['preproc_seg'])
        img = io.imread(openpath)
        segments, regions, out, labelled = superpixel_segmentation(img, superpixelsize, M, sigmaset, savepath, openpath)
        pm.update_subtask(1.0)
        pm.finish_subtask()

        pm.start_subtask(weights['generate_patches'])
        glcm_mean = np.load(glcmfeature_mean_path)
        glcm_std = np.load(glcmfeature_std_path)
        data_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        test_data1, test_data2 = generate_patches(img, labelled, regions, pad, rm, superpixelsize_r, savepath,
                                                  ctx_type, data_transform, glcm_mean, glcm_std,
                                                  progress_callback=lambda p: pm.update_subtask(p/100.0))
        pm.finish_subtask()

        pm.start_subtask(weights['classify'])
        predictions, scores = classify_patches(test_data1, test_data2, model_spec, n, batch_size=64,
                                               progress_callback=lambda p: pm.update_subtask(p/100.0),
                                               device='cuda')
        class_map = np.zeros_like(segments)
        for idx, region in enumerate(regions):
            class_map[region.coords[:, 0], region.coords[:, 1]] = predictions[idx]
        plt.imsave(os.path.join(savepath, '超像素分类结果预览图.png'), class_map,
                   cmap=cm, vmin=0, vmax=n-1)
        arr2raster(io.imread(os.path.join(savepath, '超像素分类结果预览图.png')),
                   os.path.join(savepath, '超像素分类结果地图.png'), openpath)
        pm.finish_subtask()

        pm.start_subtask(weights['cls_patch'])  # 剩余权重给切片保存
        os.makedirs(os.path.join(savepath, '切片分类结果'), exist_ok=True)
        img_cls_pad = np.pad(class_map, ((pad, pad), (pad, pad)), 'symmetric')
        total_regions = len(regions)
        # 节流：最多约 100 次更新
        update_every = max(1, total_regions // 100)
        for i, region in enumerate(regions):
            try:
                cx, cy = region.centroid
                cx, cy = int(cx), int(cy)
                cx_pad, cy_pad = cx + pad, cy + pad
                # 保存分类结果切片
                temp = img_cls_pad[cx_pad - rm:cx_pad + rm + 2, cy_pad - rm:cy_pad + rm + 2]
                plt.imsave(os.path.join(savepath, '切片分类结果', f'{ctx_type}_{i}.png'),
                           temp, cmap=cm, vmin=0, vmax=n - 1)
            except Exception as e:
                print(f"保存第 {i} 个分类切片失败: {e}")

            # 周期性上报进度
            if (i % update_every) == 0 or (i == total_regions - 1):
                frac = float(i + 1) / float(total_regions)
                try:
                    pm.update_subtask(min(max(frac, 0.0), 1.0))
                except Exception:
                    pass
        # 确保进度完成
        try:
            pm.update_subtask(1.0)
        except Exception:
            pass
        pm.finish_subtask()

        pm.start_subtask(weights['apply_mrf'])
        mrf_classes = apply_mrf_and_save(
            segments, regions, scores, n, savepath, openpath, cm,
            mrf_iterations=5, mrf_gamma=0.3, neighborhood_size=11,
            progress_callback=lambda p: pm.update_subtask(p/100.0))
        pm.finish_subtask()

        pm.start_subtask(weights['mrf_patch'])
        os.makedirs(os.path.join(savepath, '切片分类的MRF结果'), exist_ok=True)
        img_mrf_pad = np.pad(mrf_classes, ((pad, pad), (pad, pad)), 'symmetric')
        total_regions = len(regions)
        # 节流：最多约 200 次更新（可按需调整）
        update_every = max(1, total_regions // 100)
        for i, region in enumerate(regions):
            try:
                cx, cy = region.centroid
                cx, cy = int(cx), int(cy)
                cx_pad, cy_pad = cx + pad, cy + pad
                # 保存 MRF 分类结果切片
                temp_mrf = img_mrf_pad[cx_pad - rm:cx_pad + rm + 2, cy_pad - rm:cy_pad + rm + 2]
                plt.imsave(os.path.join(savepath, '切片分类的MRF结果', f'{ctx_type}_{i}.png'),
                           temp_mrf, cmap=cm, vmin=0, vmax=n)
            except Exception as e:
                # 出错时记录但继续，避免整个流程中断
                print(f"保存第 {i} 个MRF切片失败: {e}")

            # 周期性上报子任务进度（fraction 0..1）
            if (i % update_every) == 0 or (i == total_regions - 1):
                frac = float(i + 1) / float(total_regions)
                try:
                    # 保证传入的是绝对 0..1 值；有些 ProgressManager 希望绝对值
                    pm.update_subtask(min(max(frac, 0.0), 1.0))
                except Exception:
                    pass

        # overlay 回调
        try:
            cmap = mpl.cm.get_cmap(cm)
            norm = mpl.colors.Normalize(vmin=0, vmax=max(1, n - 1))
            colored = cmap(norm(mrf_classes))
            prcresult_show = (colored[:, :, :3] * 255).astype(np.uint8)
            if overlay_callback:
                overlay_callback(prcresult_show)
        except Exception as e:
            print(f"生成 overlay 失败: {e}")

        try:
            pm.update_subtask(1.0)
        except Exception:
            pass
        pm.finish_subtask()
        pm.complete()

    except Exception as e:
        try:
            pm.complete()
        except Exception:
            pass
        raise e
    finally:
        if progress_callback:
            progress_callback(100)


MODEL_REGISTRY = {
    'DenseNet161': (lambda num_classes=15: {
        'type': 'state_dict',
        'constructor': lambda: densenet161(num_classes=num_classes),
        'ckpt':  f"workdir/densenet161/new_model_1D+2DCNN5.pth",
        'strict': True,
        'two_inputs': True
    }),
    'EfficientNet-V2': (lambda num_classes=15: {
        'type': 'mmpretrain',
        'config': f'configs/efficientnet_v2/efficientnetv2-b0_8xb32_domars16k.py',
        'ckpt':  f'workdir/efficientnet_v2_domars16k_0615/epoch_180.pth',
        'two_inputs': False
    }),
    'Inception-V3': (lambda num_classes=15: {
        'type': 'mmpretrain',
        'config': f'configs/resnet/resnet50_8xb32_domars16k.py',
        'ckpt':  f'workdir/resnet_2_domars16k_0614/epoch_180.pth',
        'two_inputs': False
    }),
    'ConvNeXt-V2-T': (lambda num_classes=15: {
        'type': 'mmpretrain',
        'config': f'configs/convnext_v2/convnext-v2-tiny_32xb32_domars16k.py',
        'ckpt':  f'workdir/convnext-v2-tiny_domars16k_0627/epoch_280.pth',
        'two_inputs': False
    }),
    'MarsMapFormer': (lambda num_classes=15: {
        'type': 'mmpretrain',
        'config': f'configs/marsmapformer/marsmapformer_domars16k.py',
        'ckpt':  f'workdir/marsmapformer/epoch_200.pth',
        'two_inputs': False
    }),
    'MViTv2-T': (lambda num_classes=15: {
        'type': 'mmpretrain',
        'config': f'configs/mvit/mvitv2-tiny_8xb256_domars16k.py',
        'ckpt':  f'workdir/other/mvitv2-tiny_8xb256_domars16k_11.5/epoch_300.pth',
        'two_inputs': False
    }),
    'PoolFormer-S36': (lambda num_classes=15: {
        'type': 'mmpretrain',
        'config': f'configs/poolformer/poolformer-s36_32xb128_domars16k_2.py',
        'ckpt':  f'workdir/poolformer_domars16k_5.7/epoch_180.pth',
        'two_inputs': False
    }),
    'ResNet50': (lambda num_classes=15: {
        'type': 'mmpretrain',
        'config': f'configs/resnet/resnet50_8xb32_domars16k.py',
        'ckpt':  f'workdir/resnet_2_domars16k_0614/epoch_180.pth',
        'two_inputs': False
    }),
    'Swin-T': (lambda num_classes=15: {
        'type': 'mmpretrain',
        'config': f'configs/swin_transformer/swin-tiny_16xb64_domars16k2.py',
        'ckpt':  f'workdir/swin-tiny_domars16k_0613/epoch_195.pth',
        'two_inputs': False
    }),
    'ViT-L': (lambda num_classes=15: {
        'type': 'mmpretrain',
        'config': f'configs/vision_transformer/vit-large-p16.py',
        'ckpt':  f'workdir/vit-large-p16_8xb128-coslr-50e_domars16k_9.27/best_accuracy_top1_epoch_252.pth',
        'two_inputs': False
    }),
}


class ClassificationThread(QThread):
    # 定义所有必要的信号
    progress_updated = pyqtSignal(int)               # 进度更新信号 (0-100)
    result_ready     = pyqtSignal(np.ndarray)       # 分类结果信号（RGB overlay 等）
    error_occurred   = pyqtSignal(str)              # 错误信号
    task_completed   = pyqtSignal()                 # 任务完成信号

    def __init__(self, openpath, savepath, superpixelsize, window_size, M,
                 model_spec=None, device='cuda'):
        """
        model_spec: 可选，传入从 MODEL_REGISTRY 获取的 model_spec（mmpretrain 格式）
        device: 'cuda' 或 'cpu'
        """
        super().__init__()
        self.openpath = openpath
        self.savepath = savepath
        self.superpixelsize = superpixelsize
        self.window_size = window_size
        self.M = M
        self.model_spec = model_spec
        self.device = device
        self._stop_requested = False

    def run(self):
        try:
            # 定义回调函数
            def progress_callback(progress):
                # 通过信号发送进度更新
                self.progress_updated.emit(progress)

            def overlay_callback(image_data):
                # 通过信号发送分类结果（通常是 RGB numpy array）
                self.result_ready.emit(image_data)

            # 执行分类任务：把 model_spec 和 device 传入 slic_map_pipeline
            slic_map_pipeline(
                openpath=self.openpath,
                savepath=self.savepath,
                superpixelsize=self.superpixelsize,
                window_size=self.window_size,
                M=self.M,
                progress_callback=progress_callback,
                overlay_callback=overlay_callback,
                model_spec=self.model_spec,
                # 如果你的 slic_map_pipeline 支持 device 参数，可在此传入；若不支持可省略
                # device=self.device
            )

            # 任务完成信号
            self.task_completed.emit()


        except Exception as e:
            # 捕获完整 traceback
            tb = traceback.format_exc()
            # 尝试写入到 savepath 下的错误日志文件（若 savepath 不可写则尝试当前工作目录）
            try:
                log_dir = self.savepath if (self.savepath and os.path.isdir(self.savepath)) else os.getcwd()
                log_path = os.path.join(log_dir, "classification_error_traceback.txt")
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(tb)
            except Exception:
                # 如果写入 savepath 失败，写当前工作目录
                try:
                    log_path = os.path.join(os.getcwd(), "classification_error_traceback.txt")
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(tb)
                except Exception:
                    log_path = None
            # 发回简洁错误信息（包含日志路径）以及 traceback 的前 2000 字符作为快速预览
            preview = tb[:2000] + ("\n...[truncated]" if len(tb) > 2000 else "")
            if log_path:
                msg = f"分类错误：已将完整 traceback 保存到:\n{log_path}\n\n错误摘要（前 2000 字符）：\n{preview}"
            else:
                msg = f"分类错误（无法写入日志文件）\n错误摘要（前 2000 字符）：\n{preview}"
            # 通过信号发送（UI 的槽应弹窗或记录此字符串）
            self.error_occurred.emit(msg)
        finally:
            # 若你希望无论成功还是失败都确保线程最终发出完成信号，可在这里选择性发出
            pass

    def request_stop(self):
        """请求停止任务（当前线程需在任务中检查 self._stop_requested 才能即时响应）"""
        self._stop_requested = True

# 多线程运行
# class ClassificationThread(QThread):
#     # 定义所有必要的信号
#     progress_updated = pyqtSignal(int)  # 进度更新信号 (0-100)
#     result_ready = pyqtSignal(np.ndarray)  # 分类结果信号
#     error_occurred = pyqtSignal(str)  # 错误信号
#     task_completed = pyqtSignal()  # 任务完成信号
#     def __init__(self, openpath, savepath, ctx_type, superpixelsize, window_size, M):
#         super().__init__()
#         self.openpath = openpath
#         self.savepath = savepath
#         self.ctx_type = ctx_type
#         self.superpixelsize = superpixelsize
#         self.window_size = window_size
#         self.M = M
#         self._stop_requested = False
#
#     def run(self):
#         try:
#             # 定义回调函数
#             def progress_callback(progress):
#                 # 通过信号发送进度更新
#                 self.progress_updated.emit(progress)
#
#             def overlay_callback(image_data):
#                 # 通过信号发送分类结果
#                 self.result_ready.emit(image_data)
#
#             # 执行分类任务
#             slic_map_pipeline(
#                 openpath=self.openpath,
#                 savepath=self.savepath,
#                 ctx_type=self.ctx_type,
#                 superpixelsize=self.superpixelsize,
#                 window_size=self.window_size,
#                 M=self.M,
#                 progress_callback=progress_callback,
#                 overlay_callback=overlay_callback
#             )
#
#             # 任务完成信号
#             self.task_completed.emit()
#
#         except Exception as e:
#             # 发送错误信号
#             self.error_occurred.emit(f"分类错误: {str(e)}")
#
#     def request_stop(self):
#         """请求停止任务"""
#         self._stop_requested = True