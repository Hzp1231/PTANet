# import os
# import numpy as np
# import cv2
#
# def load_images_from_folders(visible_folder: str, thermal_folder: str, image_size=(640, 640)):
#     """
#     加载可见光图像和热红外图像并进行预处理，图像大小为 (640, 640)，每个图像为 6 通道。
#
#     :param visible_folder: 可见光图像文件夹路径
#     :param thermal_folder: 热红外图像文件夹路径
#     :param image_size: 图像的目标大小，默认为 (640, 640)
#     :return: 可见光和热红外图像对的列表
#     """
#     visible_images = sorted(os.listdir(visible_folder))
#     thermal_images = sorted(os.listdir(thermal_folder))
#
#     # 假设文件名匹配：同名可见光图像和热红外图像
#     image_pairs = []
#     for visible_img, thermal_img in zip(visible_images, thermal_images):
#         visible_img_path = os.path.join(visible_folder, visible_img)
#         thermal_img_path = os.path.join(thermal_folder, thermal_img)
#
#         # 读取可见光图像和热红外图像
#         visible = cv2.imread(visible_img_path)
#         thermal = cv2.imread(thermal_img_path)
#
#         if visible is None or thermal is None:
#             print(f"Skipping {visible_img} or {thermal_img} due to read error.")
#             continue
#
#         # Resize to target size
#         visible = cv2.resize(visible, image_size)
#         thermal = cv2.resize(thermal, image_size)
#
#         # 归一化到 [0, 1]
#         visible = visible.astype(np.float32) / 255.0
#         thermal = thermal.astype(np.float32) / 255.0
#
#         # 将可见光图像和热红外图像拼接成一个 6 通道图像
#         combined_img = np.concatenate([visible, thermal], axis=2)  # Shape: (height, width, 6)
#
#         combined_img = np.transpose(combined_img, (2, 0, 1))  # 转换为 (6, height, width)
#
#         image_pairs.append(combined_img)
#
#     return image_pairs
#
# def generate_calibration_data(visible_folder: str, thermal_folder: str, batch_size: int = 16, image_size=(640, 640)):
#     """
#     从可见光和热红外图像生成校准数据集，每次返回一个批次的校准数据。
#
#     :param visible_folder: 可见光图像文件夹路径
#     :param thermal_folder: 热红外图像文件夹路径
#     :param batch_size: 每个批次包含的图像数量
#     :param image_size: 图像的目标大小
#     :return: 每批次的 6 通道图像数据
#     """
#     # 加载图像数据
#     image_pairs = load_images_from_folders(visible_folder, thermal_folder, image_size)
#
#     # 将图像数据分成批次
#     batch = []
#     for i, image in enumerate(image_pairs):
#         batch.append(image)
#
#         # 如果批次已满，返回一个批次数据
#         if len(batch) >= batch_size:
#             yield np.array(batch)  # 返回一个批次
#             batch = []  # 清空批次
#
#     # 处理剩余的小批次
#     if batch:
#         yield np.array(batch)
#
# def save_to_bin_file(data: np.ndarray, file_path: str):
#     """
#     将 NumPy 数组保存为二进制文件。
#
#     :param data: 要保存的 NumPy 数组
#     :param file_path: 保存路径
#     """
#     with open(file_path, 'wb') as f:
#         f.write(data.astype(np.float32).tobytes())  # 将数据转换为二进制并写入文件
#
#
# # 设置可见光图像和热红外图像文件夹路径
# visible_folder = '/home/omnisky/hzp/vtuav_1.0_down/claib1/images/train'  # 替换为你的可见光图像文件夹路径
# thermal_folder = '/home/omnisky/hzp/vtuav_1.0_down/claib1/image/train'  # 替换为你的热红外图像文件夹路径
#
# # 设置保存文件的路径
# bin_file_path = 'calibration_data.bin'
#
# # 生成校准数据并保存为 bin 文件
# for batch in generate_calibration_data(visible_folder, thermal_folder, batch_size=16):
#     print(f"Saving batch with shape {batch.shape} to {bin_file_path}")
#     save_to_bin_file(batch, bin_file_path)
#     # break  # 这里只保存第一个批次，确保你可以根据需要进行调整
import numpy as np
calib_data = np.random.randn(100, 6, 640, 640).astype(np.float32)  # 生成 100 张假数据
np.save("calib_data.npy", calib_data)