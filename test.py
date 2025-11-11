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

    # 返回内存映射文件对象
    return test_data1_mem, test_data2_mem