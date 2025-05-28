import cv2
import numpy as np
import matplotlib.pyplot as plt

# 图片路径基础目录（假设在项目根目录下的 images 文件夹）
base_path = "images/"

# 图像列表
image_names = [f"HaseImage{i:02d}.jpg" for i in range(1, 6)]  # HaseImage01.jpg 到 HaseImage05.jpg

for image_name in image_names:
    image_path = base_path + image_name

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        continue

    # 转换为浮点数以便计算
    img_float = img.astype(float)

    # 步骤1: 生成暗通道 K'
    dark_channel = np.min(img_float, axis=2)

    # 步骤2: 计算全局大气光 A
    # 取暗通道中0.1%最亮像素的值作为A
    num_pixels = dark_channel.size
    num_top_pixels = int(num_pixels * 0.001)
    flat_dark_channel = dark_channel.flatten()
    dark_channel_sorted = np.sort(flat_dark_channel)[::-1]
    A = np.max(img_float[dark_channel == dark_channel_sorted[num_top_pixels-1]], axis=0)
    A = np.max(A)

    # 步骤3: 计算初始透射率 L' 并进行高斯模糊
    # 使用16x16的局部窗口取最小值
    patch_size = 16
    h, w = dark_channel.shape
    L_prime = np.zeros_like(dark_channel)
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            i_end = min(i + patch_size, h)
            j_end = min(j + patch_size, w)
            patch = dark_channel[i:i_end, j:j_end]
            L_prime[i:i_end, j:j_end] = np.min(patch)

    # 高斯模糊
    M_prime = cv2.GaussianBlur(L_prime, (151, 151), 51)

    # 步骤4: 计算透射率 t(x,y)
    omega = 0.95
    t = 1 - omega * M_prime / A
    t = np.clip(t, 0.1, 1)  # 防止t过小导致除零

    # 步骤5: 生成去雾图像 J(x,y)
    J = (img_float - (1 - t)[:, :, np.newaxis] * A) / t[:, :, np.newaxis]
    J = np.clip(J, 0, 255).astype(np.uint8)

    # 显示结果
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title(f'Original: {image_name}')
    plt.subplot(132), plt.imshow(dark_channel, cmap='gray'), plt.title('Dark Channel')
    plt.subplot(133), plt.imshow(cv2.cvtColor(J, cv2.COLOR_BGR2RGB)), plt.title('Dehazed Image')
    plt.show()

    # 保存结果
    output_path = base_path + f"dehazed_{image_name}"
    cv2.imwrite(output_path, J)
    print(f"Saved dehazed image to: {output_path}")
