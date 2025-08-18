
@jit(nopython=True)
def mrf_kernel(mrf_old, mrf_gamma, mrf, neighborhood_size):
    for r in range(mrf.shape[0]):
        for c in range(mrf.shape[1]):
            pixel_propabilities = mrf_old[
                r,
                c,
            ]
            neighbor_cnt = 0

            m = np.zeros(15)  # domars16k

            n_row_start = max(0, r - neighborhood_size)
            n_row_end = min(mrf.shape[0], r + neighborhood_size + 1)

            n_col_start = max(0, c - neighborhood_size)
            n_col_end = min(mrf.shape[1], c + neighborhood_size + 1)

            for n_row in range(n_row_start, n_row_end):
                for n_col in range(n_col_start, n_col_end):
                    if n_row != r or n_col != c:  # skip self
                        m[mrf_old[n_row, n_col, :].argmax()] += 1
                        neighbor_cnt += 1

            gibs = np.exp(-mrf_gamma * (neighbor_cnt - m))
            mrf_probabilities = gibs * pixel_propabilities
            mrf_probabilities /= np.sum(mrf_probabilities)
            mrf[
                r,
                c,
            ] = mrf_probabilities

    return mrf


def MRF(original, mrf_iterations=5, mrf_gamma=0.3, neighborhood_size=11):
    mrf_old = np.array(original)
    mrf = np.zeros(np.shape(original))
    for i in range(mrf_iterations):
        mrf = mrf_kernel(mrf_old, mrf_gamma, mrf, neighborhood_size)
        mrf_old = mrf
    return mrf


# 执行完整的分割、制图流程
def preprocess_image(openpath, ctx_type):
    img = io.imread(openpath)
    if ctx_type == "ZhuRong":
        img = gray_process(img)
    return img


def superpixel_segmentation(img, superpixelsize, M, sigmaset, savepath, openpath, cm):
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
    for i in range(np.max(segments)):
        cx, cy = regions[i].centroid
        cx, cy = int(cx), int(cy)
        # 注意：openCV 的坐标为 (x, y) == (col, row)
        cv2.circle(labelled, (cy, cx), 3, color=(1, 0, 0), thickness=-1)
        cv2.putText(labelled, str(i), (cy, cx), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(1, 0, 0), thickness=1)

    plt.imsave(os.path.join(savepath, '超像素分割标序预览图.png'), labelled)
    arr2raster((labelled * 255).astype(np.uint8), os.path.join(savepath, '超像素分割标序地图.png'), openpath)

    return segments, regions, out, labelled


def generate_patches(img, segments, regions, pad, rm, superpixelsize_r, savepath, ctx_stripe,
                     data_transform, glcmfeature_mean, glcmfeature_std):
    """
    获取图像切片，改变其格式以进行训练
    """
    os.makedirs(os.path.join(savepath, "切片数据集"), exist_ok=True)
    os.makedirs(os.path.join(savepath, "标序切片数据集"), exist_ok=True)

    img_pad = np.pad(img, ((pad, pad), (pad, pad)), 'symmetric')# 填充后的图像
    test_data1 = []
    test_data2 = []

    for i in range(np.max(segments)):
        cx, cy = regions[i].centroid
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
        glcmfeature_temp = glcm(temp2, 0, 255, 32, [2, 4, 8, 16], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
        glcmfeature_temp = np.asarray(glcmfeature_temp, dtype=np.float32)
        # 标准化（确保 mean/std 长度匹配）
        for k in range(glcmfeature_temp.shape[0]):
            glcmfeature_temp[k] = (glcmfeature_temp[k] - glcmfeature_mean[k]) / glcmfeature_std[k]
        test_data2.append(glcmfeature_temp)

    # 保存带编号的标序切片预览（在原边界图上画绿色方框）
    out = mark_boundaries(img, segments)
    out_pad = np.pad(out, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')
    for i in range(np.max(segments)):
        cx, cy = regions[i].centroid
        cx, cy = int(cx), int(cy)
        cx_pad, cy_pad = cx + pad, cy + pad
        temp = out_pad[cx_pad - rm:cx_pad + rm + 2, cy_pad - rm:cy_pad + rm + 2, :].copy()
        cv2.rectangle(temp, (rm - superpixelsize_r, rm - superpixelsize_r),
                      (rm + superpixelsize_r, rm + superpixelsize_r), color=(0, 1, 0), thickness=1)
        plt.imsave(os.path.join(savepath, '标序切片数据集', f'{ctx_stripe}_{i}.png'), temp)

    return np.asarray(test_data1, dtype=np.float32), np.asarray(test_data2, dtype=np.float32)


def classify_patches(test_data1, test_data2, model_path, num_classes, batch_size=64):
    model = densenet161(num_classes=num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    dataset1 = torch.utils.data.TensorDataset(torch.from_numpy(test_data1))
    dataset2 = torch.utils.data.TensorDataset(torch.from_numpy(test_data2[:, np.newaxis, :]))
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=False)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False)

    predictions = []
    scores = []
    with torch.no_grad():
        for (x1,), (x2,) in zip(loader1, loader2):
            x1 = x1.cuda()
            x2 = x2.cuda()
            y_hat = model(x1, x2)
            predictions.append(torch.argmax(y_hat, dim=1).cpu().numpy())
            scores.append(F.softmax(y_hat, dim=1).cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    scores = np.concatenate(scores, axis=0)
    return predictions, scores


def apply_mrf_and_save(segments, regions, scores, num_classes, savepath, openpath, cm, overlay_callback=None):
    h, w = segments.shape
    scores_map_3 = np.empty((h, w, num_classes), dtype=float)
    for j in range(np.max(segments)):
        kk = regions[j].coords
        yy = (kk[:, 0], kk[:, 1])
        for k in range(num_classes):
            scores_map_3[:, :, k][yy] = scores[j, k]

    mrf_probabilities = MRF(scores_map_3.astype(np.float64))
    mrf_classes = np.argmax(mrf_probabilities, axis=2)

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

    return mrf_classes, prcresult_show


def slic_map_pipeline(openpath, savepath, ctx_type, superpixelsize, window_size, M,
                      progress_callback=None, overlay_callback=None,
                      model_path='00save_my_train_model_x/new_model_1D+2DCNN5.pth',
                      glcmfeature_mean_path='data/final_mean.npy',
                      glcmfeature_std_path='data/final_std.npy'):
    # 进度工具
    total_steps = 7
    step = 0

    def upd(pct=None):
        nonlocal step
        if pct is None:
            step += 1
            pct = int(step / total_steps * 100)
        if progress_callback:
            progress_callback(pct)

    # 常量与超参数
    sigmaset = 5
    if window_size == 200:
        rm = 99
        pad = 100
    else:
        rm = 99
        pad = 100

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
        [140 / 255, 86 / 255, 75 / 255], # 不太一致
        [196 / 255, 156 / 255, 148 / 255],
        [227 / 255, 119 / 255, 194 / 255],
        [247 / 255, 182 / 255, 210 / 255],
        [127 / 255, 127 / 255, 127 / 255]
    ]

    hyper_params = {
        "num_classes": 15,
    }
    n = int(hyper_params["num_classes"])
    cm = mpl.colors.LinearSegmentedColormap.from_list("Custom", custom_colors, n)

    # 预处理与分割
    upd()
    img = preprocess_image(openpath, ctx_type)
    segments, regions, out, labelled = superpixel_segmentation(img, superpixelsize, M, sigmaset, savepath, openpath, cm)
    print("预处理分割完成")

    # 切片生成与特征
    upd()
    data_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    glcmfeature_mean = np.load(glcmfeature_mean_path)
    glcmfeature_std = np.load(glcmfeature_std_path)

    test_data1, test_data2 = generate_patches(img, segments, regions, pad, rm, 60, savepath, ctx_type,
                                              data_transform, glcmfeature_mean, glcmfeature_std)
    print("切片生成与特征完成")

    # 网络分类
    upd()
    predictions, scores = classify_patches(test_data1, test_data2, model_path, n)
    print("网络分类完成")

    # 从超像素还原分类图并保存
    class_map = np.zeros_like(segments)
    for j, region in enumerate(regions):
        class_map[region.coords[:, 0], region.coords[:, 1]] = predictions[j]

    plt.imsave(os.path.join(savepath, '超像素分类结果预览图.png'), class_map, cmap=cm, vmin=0, vmax=n)
    arr2raster(io.imread(os.path.join(savepath, '超像素分类结果预览图.png')),
               os.path.join(savepath, '超像素分类结果地图.png'), openpath)
    print("超像素分类图保存完成")

    upd()
    # MRF 处理并保存，同时触发 overlay_callback（如果有）
    mrf_classes, prcresult_show = apply_mrf_and_save(segments, regions, scores, n, savepath, openpath, cm, overlay_callback=overlay_callback)
    print("超像素分类图保存完成")

    upd()
    # 保存每个超像素的切片分类（原始与MRF）
    os.makedirs(os.path.join(savepath, '切片分类结果'), exist_ok=True)
    os.makedirs(os.path.join(savepath, '切片分类的MRF结果'), exist_ok=True)
    img_cls_pad = np.pad(class_map, ((pad, pad), (pad, pad)), 'symmetric')
    img_mrf_pad = np.pad(mrf_classes, ((pad, pad), (pad, pad)), 'symmetric')
    print("超像素分类MRF图保存完成")

    for i in range(np.max(segments)):
        cx, cy = regions[i].centroid
        cx, cy = int(cx), int(cy)
        cx_pad, cy_pad = cx + pad, cy + pad
        temp = img_cls_pad[cx_pad - rm:cx_pad + rm + 2, cy_pad - rm:cy_pad + rm + 2]
        plt.imsave(os.path.join(savepath, '切片分类结果', f'{ctx_type}_{i}.png'), temp, cmap=cm, vmin=0, vmax=n)

        temp_mrf = img_mrf_pad[cx_pad - rm:cx_pad + rm + 2, cy_pad - rm:cy_pad + rm + 2]
        plt.imsave(os.path.join(savepath, '切片分类的MRF结果', f'{ctx_type}_{i}.png'), temp_mrf, cmap=cm, vmin=0, vmax=n)
    print("超像素分类MRF分割结果保存完成")

    upd()
    # 记录日志
    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    filename = os.path.join(savepath, f"火星分类制图_{now_str}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        print("分割与处理完成", file=f)

