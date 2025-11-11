# coding: utf-8
# The code is written by Linghui

from skimage.feature import graycomatrix
from skimage.segmentation import slic, mark_boundaries
import datetime
from torchvision import transforms
from skimage import io
import os
from skimage.measure import regionprops
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt

import img_utils
from configs.C_model_1Dand2DCNN import densenet161
from torch.nn import functional as F
import matplotlib as mpl
from numba import jit
from img_utils import *
from PyQt5.QtCore import QThread, pyqtSignal


def image_patch(img2, slide_window, h, w):
    image = img2
    window_size = slide_window
    patch = np.zeros((slide_window, slide_window, h, w), dtype=np.uint8)

    for i in range(patch.shape[2]):
        for j in range(patch.shape[3]):
            patch[:, :, i, j] = img2[i: i + slide_window, j: j + slide_window]

    return patch


# 计算glcm特征
def calcu_glcm(img, vmin=0, vmax=255, nbit=64, step=[2], angle=[0]):
    mi, ma = vmin, vmax

    bins = np.linspace(mi, ma + 1, nbit + 1)
    img1 = np.digitize(img, bins) - 1

    glcm = graycomatrix(img1, step, angle, levels=nbit)

    return glcm


def calcu_glcm_mean(glcm, nbit=64):
    """
    calc glcm mean
    """
    mean = 0.0
    for i in range(nbit):
        for j in range(nbit):
            ss = glcm[i, j]
            mean += glcm[i, j] * i / (nbit) ** 2

    return mean


def calcu_glcm_variance(glcm, nbit=64):
    """
    calc glcm variance
    """
    mean = 0.0
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit) ** 2

    variance = 0.0
    for i in range(nbit):
        for j in range(nbit):
            variance += glcm[i, j] * (i - mean) ** 2

    return variance


def calcu_glcm_homogeneity(glcm, nbit=64):
    """
    calc glcm Homogeneity
    """
    Homogeneity = 0.0
    for i in range(nbit):
        for j in range(nbit):
            Homogeneity += glcm[i, j] / (1. + (i - j) ** 2)

    return Homogeneity


def calcu_glcm_contrast(glcm, nbit=64):
    """
    calc glcm contrast
    """
    contrast = 0.0
    for i in range(nbit):
        for j in range(nbit):
            contrast += glcm[i, j] * (i - j) ** 2

    return contrast


def calcu_glcm_dissimilarity(glcm, nbit=64):
    """
    calc glcm dissimilarity
    """
    dissimilarity = 0.0
    for i in range(nbit):
        for j in range(nbit):
            dissimilarity += glcm[i, j] * np.abs(i - j)

    return dissimilarity


def calcu_glcm_entropy(glcm, nbit=64):
    """
    calc glcm entropy
    """
    eps = 0.00001
    entropy = 0.0
    for i in range(nbit):
        for j in range(nbit):
            entropy -= glcm[i, j] * np.log10(glcm[i, j] + eps)

    return entropy


def calcu_glcm_energy(glcm, nbit=64):
    """
    calc glcm energy or second moment
    """
    energy = 0.0
    for i in range(nbit):
        for j in range(nbit):
            energy += glcm[i, j] ** 2

    return energy


def calcu_glcm_correlation(glcm, nbit=64):
    """
    calc glcm correlation (Unverified result)
    """
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


def calcu_glcm_Auto_correlation(glcm, nbit=64):
    """
    calc glcm auto correlation
    """
    Auto_correlation = 0.0
    for i in range(nbit):
        for j in range(nbit):
            Auto_correlation += glcm[i, j] * i * j

    return Auto_correlation


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


# 祝融号着陆区拼接数据需要拉伸处理
def gray_process(gray, truncated_value=0.5, maxout=255, minout=0):
    truncated_down = np.percentile(gray, truncated_value)
    truncated_up = np.percentile(gray, 100 - truncated_value)
    gray_new = ((maxout - minout) / (truncated_up - truncated_down)) * (gray - truncated_down)
    gray_new[gray_new < minout] = minout
    gray_new[gray_new > maxout] = maxout
    return np.uint8(gray_new)


def arr2raster(arr, raster_file, path):
    """
    arr:输入的mask数组 ReadAsArray()
    raster_file:输出的栅格文件路径
    prj：gdal读取的投影信息 GetProjection()，默认为空
    trans：gdal读取的几何信息GetGeoTransform()，默认为空
    """
    example = gdal.Open(path)
    prj = example.GetProjection()
    trans = example.GetGeoTransform()

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(raster_file, arr.shape[1], arr.shape[0], arr.shape[2], gdal.GDT_Byte)

    dst_ds.SetProjection(prj)
    dst_ds.SetGeoTransform(trans)

    # 将数组的各通道写入图片
    for b in range(arr.shape[2]):
        dst_ds.GetRasterBand(b + 1).WriteArray(arr[:, :, b])

    dst_ds.FlushCache()
    # print("successfully convert array to raster")


# 马尔可夫随机场后处理分割的图像
# @jit(nopython=True)
# def mrf_kernel(mrf_old, mrf_gamma, mrf, neighborhood_size):
#     H, W, C = mrf_old.shape
#     mrf_new = np.zeros((H, W, C), dtype=np.float64)
#     for h in range(H):
#         for w in range(W):
#             pixel_propabilities = mrf_old[h, w, :]
#             n_row_start = max(0, h - neighborhood_size)
#             n_row_end = min(mrf.shape[0], h + neighborhood_size + 1)
#             n_col_start = max(0, w - neighborhood_size)
#             n_col_end = min(mrf.shape[1], w + neighborhood_size + 1)
#             neighbor_cnt = 0
#             m = np.zeros(C, dtype=np.float64)  # domars16k
#             for n_row in range(n_row_start, n_row_end):
#                 for n_col in range(n_col_start, n_col_end):
#                     if n_row != h or n_col != w:  # skip self
#                         lab = np.argmax(mrf_old[n_row, n_col, :])
#                         m[lab] += 1.0
#                         neighbor_cnt += 1
#             gibs = np.exp(-mrf_gamma * (neighbor_cnt - m))
#             mrf_probabilities = gibs * pixel_propabilities
#             mrf_probabilities /= np.sum(mrf_probabilities)
#             mrf_new[h, w, :] = mrf_probabilities
#     return mrf
@jit(nopython=True)
def mrf_kernel(mrf_old, mrf_gamma, neighborhood_size):
    """
    单次 MRF 迭代核（numba-friendly）。
    输入:
      mrf_old: H x W x C (float64) - 每像素每类的概率
      mrf_gamma: float
      neighborhood_size: int (radius)
    返回:
      mrf_new: H x W x C (float64)
    """
    H, W, C = mrf_old.shape
    mrf_new = np.zeros((H, W, C), dtype=np.float64)

    for r in range(H):
        for c in range(W):
            # 当前像素的概率向量
            pixel_probs = mrf_old[r, c, :]  # length C

            # 邻域边界
            r0 = r - neighborhood_size
            if r0 < 0:
                r0 = 0
            r1 = r + neighborhood_size + 1
            if r1 > H:
                r1 = H

            c0 = c - neighborhood_size
            if c0 < 0:
                c0 = 0
            c1 = c + neighborhood_size + 1
            if c1 > W:
                c1 = W

            neighbor_cnt = 0
            votes = np.zeros(C, dtype=np.float64)

            # 计算邻居投票（用邻居的 argmax）
            for nr in range(r0, r1):
                for nc in range(c0, c1):
                    if nr == r and nc == c:
                        continue
                    # 手写 argmax（numba 兼容）
                    maxv = mrf_old[nr, nc, 0]
                    lab = 0
                    for kk in range(1, C):
                        val = mrf_old[nr, nc, kk]
                        if val > maxv:
                            maxv = val
                            lab = kk
                    votes[lab] += 1.0
                    neighbor_cnt += 1

            # 计算 gibs（每个类别的权重）
            if neighbor_cnt == 0:
                gibs = np.ones(C, dtype=np.float64)
            else:
                # gibs_k = exp(-gamma*(neighbor_cnt - votes_k))
                gibs = np.exp(-mrf_gamma * (neighbor_cnt - votes))

            # 结合并归一化，保护分母
            combined = np.empty(C, dtype=np.float64)
            s = 0.0
            for kk in range(C):
                combined[kk] = gibs[kk] * pixel_probs[kk]
                s += combined[kk]
            if s <= 0.0:
                # 退化处理：归一化原始 pixel_probs（若也为 0，则均匀分布）
                s2 = 0.0
                for kk in range(C):
                    s2 += pixel_probs[kk]
                if s2 <= 0.0:
                    # 全为 0 -> 均匀分布
                    for kk in range(C):
                        mrf_new[r, c, kk] = 1.0 / C
                else:
                    for kk in range(C):
                        mrf_new[r, c, kk] = pixel_probs[kk] / s2
            else:
                for kk in range(C):
                    mrf_new[r, c, kk] = combined[kk] / s
    return mrf_new


# 修改MRF进mrf_kernel函数中
# def MRF(original, mrf_iterations=5, mrf_gamma=0.3, neighborhood_size=11):
#     mrf_old = np.array(original)
#     mrf = np.zeros(np.shape(original))
#     for i in range(mrf_iterations):
#         mrf = mrf_kernel(mrf_old, mrf_gamma, mrf, neighborhood_size)
#         mrf_old = mrf
#     return mrf


# 执行完整的分割、制图流程
def preprocess_image(openpath, ctx_type):
    img = io.imread(openpath)
    if ctx_type == "ZhuRong":
        img = gray_process(img)
    return img


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


def generate_patches(img, segments, regions, pad, rm, superpixelsize_r, savepath, ctx_stripe,
                     data_transform, glcmfeature_mean, glcmfeature_std,
                     progress_callback=None):
    """
    获取图像切片，改变其格式以进行训练
    progress_callback用于进度条的更新
    """
    os.makedirs(os.path.join(savepath, "切片数据集"), exist_ok=True)
    os.makedirs(os.path.join(savepath, "标序切片数据集"), exist_ok=True)

    img_pad = np.pad(img, ((pad, pad), (pad, pad)), 'symmetric')# 填充后的图像
    N = len(regions)
    test_data1 = []
    test_data2 = []

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
        test_data1.append(np.asarray(temp1, dtype=np.float32))
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
        test_data2.append(glcmfeature_temp)

        # 进度回调（按批次减少 UI 负担）
        units_done += 1
        if (i % report_every_A == 0) or (i == N - 1):
            report()

        # for k in range(glcmfeature_temp.shape[0]):
        #     glcmfeature_temp[k] = (glcmfeature_temp[k] - glcmfeature_mean[k]) / glcmfeature_std[k]
        # test_data2.append(glcmfeature_temp)

    # 保存带编号的标序切片预览（在原边界图上画绿色方框）
    out = mark_boundaries(img, segments)
    out_pad = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')

    for i, region in enumerate(regions):
        cx, cy = region.centroid
        cx, cy = int(cx), int(cy)
        cx_pad, cy_pad = cx + pad, cy + pad
        temp = out_pad[cx_pad - rm:cx_pad + rm + 2, cy_pad - rm:cy_pad + rm + 2, :]
        cv2.rectangle(temp, (rm - superpixelsize_r, rm - superpixelsize_r),
                      (rm + superpixelsize_r, rm + superpixelsize_r), color=(0, 1, 0), thickness=1)
        plt.imsave(os.path.join(savepath, '标序切片数据集', f'{ctx_stripe}_{i}.png'), temp)

        units_done += 1
        if (i % report_every_B == 0) or (i == N - 1):
            report()

    # 最终确保进度 100%
    if progress_callback:
        progress_callback(100)

    return np.asarray(test_data1, dtype=np.float32), np.asarray(test_data2, dtype=np.float32)


def classify_patches(test_data1, test_data2, model_path, num_classes, batch_size=64,
                     progress_callback=None):
    model = densenet161(num_classes=num_classes)
    assert isinstance(model_path, object)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    dataset1 = torch.utils.data.TensorDataset(torch.from_numpy(test_data1))
    dataset2 = torch.utils.data.TensorDataset(torch.from_numpy(test_data2[:, np.newaxis, :]))
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=False)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False)

    total_batches = max(len(loader1), len(loader2))
    predictions = []
    scores = []
    with torch.no_grad():
        for b_idx, ((x1,), (x2,)) in enumerate(zip(loader1, loader2)):
            x1 = x1.cuda()
            x2 = x2.cuda()
            y_hat = model(x1, x2)
            predictions.append(torch.argmax(y_hat, dim=1).cpu().numpy())
            scores.append(F.softmax(y_hat, dim=1).cpu().numpy())

            if progress_callback:
                pct = int(round((b_idx + 1) / float(total_batches) * 100))
                progress_callback(pct)

    predictions = np.concatenate(predictions, axis=0)
    scores = np.concatenate(scores, axis=0)

    if progress_callback:
        progress_callback(100)

    return predictions, scores


def apply_mrf_and_save(segments, regions, scores, num_classes, savepath, openpath, cm,
                       mrf_iterations=5, mrf_gamma=0.3, neighborhood_size=11,
                       progress_callback=None, overlay_callback=None):
    h, w = segments.shape
    scores_map_3 = np.empty((h, w, num_classes), dtype=np.float64)
    for j, region in enumerate(regions):
        yy = (region.coords[:, 0], region.coords[:, 1])
        for k in range(num_classes):
            scores_map_3[yy] = scores[j, :]

    # mrf_probabilities = MRF(scores_map_3.astype(np.float64),
    #                         mrf_iterations, mrf_gamma, neighborhood_size)
    mrf_prob = scores_map_3.astype(np.float64)
    for it in range(mrf_iterations):
        mrf_prob = mrf_kernel(mrf_prob, float(mrf_gamma), int(neighborhood_size))

        if progress_callback:
            pct = int(round((it + 1) / float(mrf_iterations) * 100))
            progress_callback(pct)

    mrf_classes = np.argmax(mrf_prob, axis=2)

    # 保存MRF结果预览与地理图
    plt.imsave(os.path.join(savepath, 'MRF分类结果预览图.png'), mrf_classes, cmap=cm, vmin=0, vmax=num_classes)
    arr2raster(io.imread(os.path.join(savepath, 'MRF分类结果预览图.png')),
               os.path.join(savepath, 'MRF分类结果地图.png'), openpath)

    # 生成用于 overlay 的 RGB 预览（使用同样的 colormap，提取前三个通道）
    cmap = mpl.cm.get_cmap(cm)
    norm = mpl.colors.Normalize(vmin=0, vmax=max(1, num_classes - 1))
    colored = cmap(norm(mrf_classes))  # RGBA float 0..1
    prcresult_show = (colored[:, :, :3] * 255).astype(np.uint8)

    # 如果有 overlay_callback，回传 RGB 图像
    if overlay_callback:
        overlay_callback(prcresult_show)

    if progress_callback:
        progress_callback(100)

    return mrf_classes


def slic_map_pipeline(openpath, savepath, ctx_type, superpixelsize, window_size, M,
                      progress_callback=None, overlay_callback=None,
                      model_path='00save_my_train_model_x/new_model_1D+2DCNN5.pth',
                      glcmfeature_mean_path='data/final_mean.npy',
                      glcmfeature_std_path='data/final_std.npy'):
    if progress_callback:
        progress_callback(0)

    pm = img_utils.ProgressManager(progress_callback)

    weights = {
        'preproc_seg': 0.05,
        'generate_patches': 0.25,
        'classify': 0.45,
        'apply_mrf': 0.2,
        'save_finalize': 0.05
    }

    # 常量与超参数
    sigmaset = 5
    if window_size == 200:
        rm = 99
        pad = 100
    else:
        rm = 99
        pad = 100
    superpixelsize_r = 60

    os.makedirs(savepath, exist_ok=True)

    custom_colors = [
        [31 / 255, 119 / 255, 180 / 255],# 曲线型沙丘
        [174 / 255, 199 / 255, 232 / 255],# 直线型沙丘
        [255 / 255, 127 / 255, 14 / 255],# 橙色，悬崖
        [255 / 255, 187 / 255, 120 / 255],# 撞击坑
        [44 / 255, 160 / 255, 44 / 255],# 斜坡条纹
        [152 / 255, 223 / 255, 138 / 255],# 沟槽
        [214 / 255, 39 / 255, 40 / 255],# 冲沟
        [255 / 255, 152 / 255, 150 / 255],# 滑坡
        [148 / 255, 103 / 255, 189 / 255],# 混合
        [197 / 255, 176 / 255, 213 / 255],# 山脊
        [140 / 255, 86 / 255, 75 / 255],  # 粗糙# 不太一致
        [196 / 255, 156 / 255, 148 / 255],# 土丘
        [227 / 255, 119 / 255, 194 / 255],# 撞击坑群
        [247 / 255, 182 / 255, 210 / 255],# 光滑形貌
        [127 / 255, 127 / 255, 127 / 255]# 纹理
    ]

    hyper_params = {
        "num_classes": 15,
    }
    n = int(hyper_params["num_classes"])
    cm = mpl.colors.LinearSegmentedColormap.from_list("Custom", custom_colors, n)

    try:
        # Stage 1: preprocess + segmentation
        pm.start_subtask(weights['preproc_seg'])
        img = preprocess_image(openpath, ctx_type)
        segments, regions, out, labelled = superpixel_segmentation(img, superpixelsize, M, sigmaset, savepath, openpath)
        pm.update_subtask(1.0)
        pm.finish_subtask()

        # Stage 2: generate patches
        pm.start_subtask(weights['generate_patches'])
        glcm_mean = np.load(glcmfeature_mean_path)
        glcm_std = np.load(glcmfeature_std_path)
        data_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_data1, test_data2 = generate_patches(img, segments, regions, pad, rm, superpixelsize_r, savepath,
                                                ctx_type, data_transform, glcm_mean, glcm_std,
                                                progress_callback=lambda p: pm.update_subtask(p/100.0))
        pm.finish_subtask()

        # Stage 3: classification
        pm.start_subtask(weights['classify'])
        predictions, scores = classify_patches(test_data1, test_data2, model_path, n, batch_size=64,
                                         progress_callback=lambda p: pm.update_subtask(p/100.0))
        pm.finish_subtask()

        date1 = datetime.datetime.now()
        # restore无MRF的分类结果
        class_map = np.zeros_like(segments)
        for idx, region in enumerate(regions):
            class_map[region.coords[:, 0], region.coords[:, 1]] = predictions[idx]
        plt.imsave(os.path.join(savepath, '超像素分类结果预览图.png'), class_map,
                   cmap=cm, vmin=0, vmax=n-1)
        arr2raster(io.imread(os.path.join(savepath, '超像素分类结果预览图.png')),
                   os.path.join(savepath, '超像素分类结果地图.png'), openpath)
        date2 = datetime.datetime.now()
        print((date2-date1).seconds)

        # Stage 4: apply MRF
        pm.start_subtask(weights['apply_mrf'])
        mrf_classes = apply_mrf_and_save(
            segments, regions, scores, n, savepath, openpath, cm,
            mrf_iterations=5, mrf_gamma=0.3, neighborhood_size=11,
            progress_callback=lambda p: pm.update_subtask(p/100.0),
            overlay_callback=overlay_callback)
        pm.finish_subtask()

        # 保存分类的切片结果
        os.makedirs(os.path.join(savepath, '切片分类结果'), exist_ok=True)
        os.makedirs(os.path.join(savepath, '切片分类的MRF结果'), exist_ok=True)
        img_cls_pad = np.pad(class_map, ((pad, pad), (pad, pad)), 'symmetric')
        img_mrf_pad = np.pad(mrf_classes, ((pad, pad), (pad, pad)), 'symmetric')
        for i, region in enumerate(regions):
            cx, cy = region.centroid
            cx, cy = int(cx), int(cy)
            cx_pad, cy_pad = cx + pad, cy + pad
            temp = img_cls_pad[cx_pad - rm:cx_pad + rm + 2, cy_pad - rm:cy_pad + rm + 2]
            plt.imsave(os.path.join(savepath, '切片分类结果', f'{ctx_type}_{i}.png'), temp, cmap=cm, vmin=0, vmax=n-1)

            temp_mrf = img_mrf_pad[cx_pad - rm:cx_pad + rm + 2, cy_pad - rm:cy_pad + rm + 2]
            plt.imsave(os.path.join(savepath, '切片分类的MRF结果', f'{ctx_type}_{i}.png'), temp_mrf, cmap=cm, vmin=0,
                       vmax=n)

        # Stage 5: finalize saves (already mostly done in apply_mrf_and_save)
        pm.start_subtask(weights['save_finalize'])
        pm.update_subtask(1.0)
        pm.finish_subtask()

        pm.complete()


    except Exception as e:
        try:
            pm.complete()
        except Exception:
            pass
        raise e
    finally:
        # 最终确保进度为100%
        if progress_callback:
            progress_callback(100)


# 多线程运行
class ClassificationThread(QThread):
    # 定义所有必要的信号
    progress_updated = pyqtSignal(int)  # 进度更新信号 (0-100)
    result_ready = pyqtSignal(np.ndarray)  # 分类结果信号
    error_occurred = pyqtSignal(str)  # 错误信号
    task_completed = pyqtSignal()  # 任务完成信号
    def __init__(self, openpath, savepath, ctx_type, superpixelsize, window_size, M):
        super().__init__()
        self.openpath = openpath
        self.savepath = savepath
        self.ctx_type = ctx_type
        self.superpixelsize = superpixelsize
        self.window_size = window_size
        self.M = M
        self._stop_requested = False

    def run(self):
        try:
            # 定义回调函数
            def progress_callback(progress):
                # 通过信号发送进度更新
                self.progress_updated.emit(progress)

            def overlay_callback(image_data):
                # 通过信号发送分类结果
                self.result_ready.emit(image_data)

            # 执行分类任务
            slic_map_pipeline(
                openpath=self.openpath,
                savepath=self.savepath,
                ctx_type=self.ctx_type,
                superpixelsize=self.superpixelsize,
                window_size=self.window_size,
                M=self.M,
                progress_callback=progress_callback,
                overlay_callback=overlay_callback
            )

            # 任务完成信号
            self.task_completed.emit()

        except Exception as e:
            # 发送错误信号
            self.error_occurred.emit(f"分类错误: {str(e)}")

    def request_stop(self):
        """请求停止任务"""
        self._stop_requested = True