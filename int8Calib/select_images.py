import os
from glob import glob
import shutil

# 根目录
data_root = "/home/omnisky/hzp/vtuav_1.0_down"

# 子目录
visible_dir = os.path.join(data_root, "images/train")
thermal_dir = os.path.join(data_root, "image/train")
label_dir = os.path.join(data_root, "labels/train")

# 输出路径
output_root = "/home/omnisky/hzp/vtuav_1.0_down/claib1"
output_visible = os.path.join(output_root, "images/train")
output_thermal = os.path.join(output_root, "image/train")
output_labels = os.path.join(output_root, "labels/train")

# 创建输出目录
os.makedirs(output_visible, exist_ok=True)
os.makedirs(output_thermal, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# 获取所有文件名
image_filenames = sorted([os.path.basename(f) for f in glob(os.path.join(visible_dir, "*.jpg"))])

# 每20帧选一帧
selected_filenames = image_filenames[::20]

# 拷贝配对文件到输出目录
for fname in selected_filenames:
    src_visible = os.path.join(visible_dir, fname)
    src_thermal = os.path.join(thermal_dir, fname)
    src_label   = os.path.join(label_dir, fname.replace(".jpg", ".txt"))

    dst_visible = os.path.join(output_visible, fname)
    dst_thermal = os.path.join(output_thermal, fname)
    dst_label   = os.path.join(output_labels, fname.replace(".jpg", ".txt"))

    shutil.copy(src_visible, dst_visible)
    shutil.copy(src_thermal, dst_thermal)
    shutil.copy(src_label, dst_label)

print(f"✅ 已成功保存 {len(selected_filenames)} 对配对图像到 {output_root}/")