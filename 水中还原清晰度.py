import cv2
import numpy as np
import matplotlib.pyplot as plt

# 图片路径列表（假设在项目根目录下的 images 文件夹）
img_paths = [
    'images/input_img01.jpg',
    'images/input_img02.jpg',
    'images/input_img03.jpg',
    'images/input_img04.jpg',
    'images/input_img05.jpg'
]

# 参考颜色数组
bgr_7m = np.array([
    [138, 91, 0],
    [15, 71, 182],
    [134, 129, 168],
    [42, 220, 0],
    [205, 254, 2],
    [12, 254, 105],
    [208, 255, 103],
    [43, 98, 5],
    [19, 59, 1]
], dtype=np.float32)

bgr_1m = np.array([
    [218, 60, 36],
    [48, 69, 255],
    [227, 109, 252],
    [70, 184, 36],
    [255, 252, 44],
    [28, 247, 255],
    [254, 255, 250],
    [73, 78, 27],
    [37, 52, 8]
], dtype=np.float32)

def color_restore(img, ref_src=bgr_7m, ref_dst=bgr_1m):
    h, w, _ = img.shape
    restored = np.zeros_like(img, dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            pixel = img[y, x].astype(np.float32)

            dists = np.linalg.norm(ref_src - pixel, axis=1)

            weights = 1.0 / (dists + 1e-10)
            weights /= np.sum(weights)

            new_pixel = np.zeros(3, dtype=np.float32)
            for channel in range(3):
                numerator = np.sum(weights * ref_dst[:, channel])
                denominator = np.sum(weights * ref_src[:, channel])

                new_pixel[channel] = pixel[channel] * (numerator / denominator)

            restored[y, x] = np.clip(new_pixel, 0, 255).astype(np.uint8)

    return restored

for path in img_paths:
    img = cv2.imread(path)

    if img is None:
        print(f"Image not found: {path}")
        continue

    restored = color_restore(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    restored_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.imshow(restored_rgb)
    plt.title(f'Restored: {os.path.basename(path)}')
    plt.axis('off')
    plt.show()
