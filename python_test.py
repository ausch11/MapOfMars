import json
import os, re
from datetime import datetime

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from configs.C_model_1Dand2DCNN import densenet161

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
from torch.nn import functional as F
from mmpretrain.apis import init_model
from mmpretrain.utils import register_all_modules
from skimage.feature import graycomatrix

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
    spec_factory = model_spec['DenseNet161']
    spec = spec_factory(num_classes=15)
    model_type = spec.get('type')
    two_inputs = bool(spec.get('two_inputs'))

    # 选择 device：优先使用用户指定，但如果 cuda 不可用则降到 cpu
    device_t = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    if model_type == 'mmpretrain':
        # 原来的 mmpretrain 加载逻辑
        cfg = spec.get('config')
        ckpt = spec.get('ckpt')
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
        constructor = spec.get('constructor')
        ckpt = spec.get('ckpt')
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

        def predict_fn(x):
            # 两路输入
            out = model(x[0].to(device_t), x[1].to(device_t))
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
    if test_data1 is None or test_data2 is None:
        raise ValueError("输入分类数据为 None")

    test_data1 = np.asarray(test_data1, dtype=np.float32)
    test_data2 = np.asarray(test_data2, dtype=np.float32)

    N1 = int(test_data1.shape[0])
    N2 = int(test_data2.shape[0])
    if N1 != N2:
        raise ValueError(f"样本数不一致: test_data1={N1}, test_data2={N2}")

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

    return predictions.astype(np.int32), scores.astype(np.float32)

def _index_from_filename(fn: str):
    m = re.search(r'_(\d+)(?=\.[^.]+$)', fn)
    if m:
        return int(m.group(1))
    m2 = re.search(r'(\d+)', fn)
    if m2:
        return int(m2.group(1))
    return -1

def compute_and_save_patches_and_glcm(image_dir,
                                      cache_dir,
                                      glcmfeature_mean_path,
                                      glcmfeature_std_path,
                                      transform=None,
                                      superpixelsize_r=60,
                                      nbit=32,
                                      step_list=(2,4,8,16),
                                      angle_list=None,
                                      ext_allow=('.png','.jpg','.jpeg','.tif','.tiff'),
                                      progress_callback=None,
                                      overwrite=False):
    """
    读取 image_dir 中切片图片（按文件名数字排序），对每张：
      - 计算 GLCM 特征并标准化（使用 glcmfeature_mean_path / glcmfeature_std_path）
      - 计算并保存 transform 后的 tensor (C,H,W) 为 numpy
    缓存结构 (cache_dir):
      cache_dir/
        manifest.json            # 文件名顺序 & metadata
        glcm_npy/                # per-sample feature files <basename>.npy
        patches_npy/             # per-sample image tensors <basename>.npy (可选)
        test_data1.npy           # 聚合后的 images (N,C,H,W)   (可选)
        test_data2.npy           # 聚合后的 features (N,F)    (可选)
    Returns:
      test_data1 (N,C,H,W) numpy float32, test_data2 (N,F) numpy float32, files_sorted list
    """
    transform = transform or DEFAULT_TRANSFORM
    if angle_list is None:
        angle_list = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    os.makedirs(cache_dir, exist_ok=True)
    glcm_dir = os.path.join(cache_dir, "glcm_npy")
    patches_dir = os.path.join(cache_dir, "patches_npy")
    os.makedirs(glcm_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)

    manifest_path = os.path.join(cache_dir, "manifest.json")
    agg_img_path = os.path.join(cache_dir, "test_data1.npy")
    agg_feat_path = os.path.join(cache_dir, "test_data2.npy")

    # load mean/std
    glcmfeature_mean = np.load(glcmfeature_mean_path)
    glcmfeature_std = np.load(glcmfeature_std_path)

    # find image files & sort
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(ext_allow)]
    if not files:
        raise FileNotFoundError(f"No image files in {image_dir}")
    files_sorted = sorted(files, key=_index_from_filename)
    N = len(files_sorted)

    # If aggregated cache exists and not overwrite -> load and return early
    if (not overwrite) and os.path.exists(agg_img_path) and os.path.exists(agg_feat_path) and os.path.exists(manifest_path):
        # load manifest to ensure order
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("source_dir") == os.path.abspath(image_dir):
            test_data1 = np.load(agg_img_path)
            test_data2 = np.load(agg_feat_path)
            return test_data1.astype(np.float32), test_data2.astype(np.float32), files_sorted

    # Otherwise compute per-sample (but skip per-sample if already exists and not overwrite)
    imgs = []
    feats = []
    total = N
    report_every = max(1, total // 100)

    for i, fn in enumerate(files_sorted):
        base = os.path.splitext(fn)[0]
        img_path = os.path.join(image_dir, fn)
        glcm_np_path = os.path.join(glcm_dir, base + ".npy")
        patch_np_path = os.path.join(patches_dir, base + ".npy")

        # If both per-sample files exist and not overwrite -> load them
        if (not overwrite) and os.path.exists(glcm_np_path) and os.path.exists(patch_np_path):
            feat = np.load(glcm_np_path).astype(np.float32)
            patch = np.load(patch_np_path).astype(np.float32)
            feats.append(feat)
            imgs.append(patch)
        else:
            # read image
            im = Image.open(img_path)
            im_gray = im.convert('L')
            arr_gray = np.asarray(im_gray, dtype=np.float32)
            H, W = arr_gray.shape
            cx = H // 2
            cy = W // 2
            r = int(superpixelsize_r)
            r0 = max(0, cx - r)
            r1 = min(H, cx + r + 1)
            c0 = max(0, cy - r)
            c1 = min(W, cy + r + 1)
            temp2 = arr_gray[r0:r1, c0:c1]
            expected_h = 2 * r + 1
            expected_w = 2 * r + 1
            if temp2.shape[0] != expected_h or temp2.shape[1] != expected_w:
                pad_top = max(0, r - cx)
                pad_bottom = max(0, (cx + r) - (H - 1))
                pad_left = max(0, r - cy)
                pad_right = max(0, (cy + r) - (W - 1))
                temp2 = np.pad(temp2, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='symmetric')

            # compute glcm features (依赖你模块中的 glcm() 函数)
            glcm_feat = glcm(temp2, 0, 255, nbit, list(step_list), angle_list)
            glcm_feat = np.asarray(glcm_feat, dtype=np.float32)
            glcm_len = glcm_feat.shape[0]
            if glcmfeature_mean.shape[0] < glcm_len or glcmfeature_std.shape[0] < glcm_len:
                raise RuntimeError(f"GLCM mean/std length smaller than computed feature length {glcm_len}")
            glcm_feat = (glcm_feat - glcmfeature_mean[:glcm_len]) / (glcmfeature_std[:glcm_len] + 1e-12)

            # prepare image tensor via transform -> numpy (C,H,W)
            im_rgb = im.convert('RGB')
            t = transform(im_rgb)  # torch.Tensor CxHxW
            patch_arr = np.asarray(t, dtype=np.float32)

            # save per-sample files
            np.save(glcm_np_path, glcm_feat)
            np.save(patch_np_path, patch_arr)

            feats.append(glcm_feat)
            imgs.append(patch_arr)

        # progress callback
        if progress_callback and ((i % report_every == 0) or (i == total - 1)):
            pct = int(round((i + 1) / float(total) * 100))
            progress_callback(pct)

    # Stack into aggregate arrays and write aggregate .npy
    test_data1 = np.stack(imgs, axis=0).astype(np.float32)  # (N, C, H, W)
    test_data2 = np.stack(feats, axis=0).astype(np.float32) # (N, F)

    np.save(agg_img_path, test_data1)
    np.save(agg_feat_path, test_data2)

    # write manifest
    manifest = {
        "source_dir": os.path.abspath(image_dir),
        "n_samples": len(files_sorted),
        "files": files_sorted,
        "agg_img": os.path.basename(agg_img_path),
        "agg_feat": os.path.basename(agg_feat_path),
        "generated_at": datetime.now().isoformat()
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return test_data1, test_data2, files_sorted


def load_cached_patches_and_glcm(cache_dir):
    """
    更宽松的缓存加载函数：
      - 优先读取 test_data1.npy 与 test_data2.npy（即便 manifest 不存在也能加载）
      - 如果聚合文件不存在，则尝试按 per-sample 目录 glcm_npy/ 和 patches_npy/ 读取
      - 返回 (test_data1, test_data2, files_sorted)
         - 如果只有 test_data1 而没有 test_data2，则返回 (test_data1, None, files_sorted_or_indices)
    抛错情形:
      - cache_dir 不存在或既没有聚合文件也没有 per-sample 文件
    """
    manifest_path = os.path.join(cache_dir, "manifest.json")
    agg_img_path = os.path.join(cache_dir, "test_data1.npy")
    agg_feat_path = os.path.join(cache_dir, "test_data2.npy")
    glcm_dir = os.path.join(cache_dir, "glcm_npy")
    patches_dir = os.path.join(cache_dir, "patches_npy")

    # 1) If both aggregate npy present -> load them (manifest optional)
    if os.path.exists(agg_img_path) and os.path.exists(agg_feat_path):
        test_data1 = np.load(agg_img_path).astype(np.float32)
        test_data2 = np.load(agg_feat_path).astype(np.float32)
        files_sorted = None
        # try to recover files_sorted from manifest if exists
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                files_sorted = manifest.get("files", None)
            except Exception:
                files_sorted = None
        # if no manifest, attempt to infer from per-sample folder names
        if files_sorted is None:
            if os.path.isdir(patches_dir):
                fns = sorted([f for f in os.listdir(patches_dir) if f.lower().endswith('.npy')])
                if fns:
                    # strip extensions to get basenames (original image filenames may be basename)
                    files_sorted = [os.path.splitext(fn)[0] for fn in fns]
                else:
                    files_sorted = list(range(test_data1.shape[0]))
            else:
                files_sorted = list(range(test_data1.shape[0]))
        return test_data1, test_data2, files_sorted

    # 2) If only aggregate image present
    if os.path.exists(agg_img_path) and not os.path.exists(agg_feat_path):
        test_data1 = np.load(agg_img_path).astype(np.float32)
        test_data2 = None
        files_sorted = None
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                files_sorted = manifest.get("files", None)
            except Exception:
                files_sorted = None
        if files_sorted is None:
            # fallback to indices
            files_sorted = list(range(test_data1.shape[0]))
        return test_data1, test_data2, files_sorted

    # 3) If only aggregate feature present
    if os.path.exists(agg_feat_path) and not os.path.exists(agg_img_path):
        test_data1 = None
        test_data2 = np.load(agg_feat_path).astype(np.float32)
        files_sorted = None
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                files_sorted = manifest.get("files", None)
            except Exception:
                files_sorted = None
        if files_sorted is None:
            files_sorted = list(range(test_data2.shape[0]))
        return test_data1, test_data2, files_sorted

    # 4) No aggregate npy -> try per-sample directories
    if os.path.isdir(glcm_dir) and os.path.isdir(patches_dir):
        feat_files = sorted([f for f in os.listdir(glcm_dir) if f.lower().endswith('.npy')])
        patch_files = sorted([f for f in os.listdir(patches_dir) if f.lower().endswith('.npy')])
        if len(feat_files) == 0 and len(patch_files) == 0:
            # nothing to load
            raise FileNotFoundError(f"No aggregate files and no per-sample files found in {cache_dir}")
        # try to match by basename order; if mismatch, use smaller common set
        basenames_feat = [os.path.splitext(f)[0] for f in feat_files]
        basenames_patch = [os.path.splitext(f)[0] for f in patch_files]
        common = [b for b in basenames_patch if b in basenames_feat]
        if not common:
            # if no common basenames, try sorted order intersection
            N = min(len(basenames_patch), len(basenames_feat))
            common = basenames_patch[:N]
        feats = [np.load(os.path.join(glcm_dir, b + ".npy")).astype(np.float32) for b in common]
        patches = [np.load(os.path.join(patches_dir, b + ".npy")).astype(np.float32) for b in common]
        test_data1 = np.stack(patches, axis=0)
        test_data2 = np.stack(feats, axis=0)
        files_sorted = common
        return test_data1, test_data2, files_sorted

    # 5) Nothing found
    raise FileNotFoundError(f"No cache found in {cache_dir} (no aggregate npy and no per-sample glcm_npy/patches_npy). "
                            f"Run compute_and_save_patches_and_glcm(...) first.")

image_dir = r"E:\TEMP\切片数据集"
cache_dir = r"E:\TEMP\npy" # 我建议单独建一个缓存目录
glcm_mean = r'data/final_mean.npy'
glcm_std = r'data/final_std.npy'

# 读取并计算（若想在 GUI 进度条里显示，传一个回调）
def progress_cb(p):
    print("Loading & GLCM progress:", p, "%")

# test_data1, test_data2, files_sorted = compute_and_save_patches_and_glcm(
#     image_dir=image_dir,
#     cache_dir=cache_dir,
#     glcmfeature_mean_path=glcm_mean,
#     glcmfeature_std_path=glcm_std,
#     transform=DEFAULT_TRANSFORM,
#     superpixelsize_r=60,
#     progress_callback=lambda p: print("compute & save progress:", p, "%"),
#     overwrite=False
# )
model_spec = {
    'DenseNet161': (lambda num_classes=15: {
        'type': 'state_dict',
        'constructor': lambda: densenet161(num_classes=num_classes),
        'ckpt':  f"E:/a_GUI_rs/SegMars/00save_my_train_model_x/new_model_1D+2DCNN5.pth",
        'strict': True,
        'two_inputs': True
    })}

test_data1, test_data2, files_sorted = load_cached_patches_and_glcm(cache_dir)
# 直接把 test_data1/test_data2 传给 classify_patches(...)
predictions, scores = classify_patches(test_data1, test_data2, model_spec, num_classes=15, batch_size=64)
print(predictions, scores)



