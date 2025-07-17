# import os
# import numpy as np
# import tensorrt as trt
# from utils import load_image_pair
#
#
# class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
#     def __init__(self, rgb_dir, thermal_dir, input_shape, cache_file="calib_cache.bin", batch_size=1):
#         super().__init__()
#         self.input_shape = input_shape  # (C, H, W)
#         self.batch_size = batch_size
#         self.cache_file = cache_file
#
#         # 加载图像路径
#         self.rgb_paths = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.png'))])
#         self.thermal_paths = sorted([os.path.join(thermal_dir, f) for f in os.listdir(thermal_dir) if f.lower().endswith(('.jpg', '.png'))])
#         assert len(self.rgb_paths) == len(self.thermal_paths), "Mismatch in number of RGB and thermal images."
#
#         # 加载所有样本
#         self.data = self._load_batches()
#         self.current_index = 0
#
#     def _load_batches(self):
#         H, W = self.input_shape[1], self.input_shape[2]
#         batches = []
#         for rgb_path, thermal_path in zip(self.rgb_paths, self.thermal_paths):
#             # 加载并堆叠 RGB + 热红外图像，返回 (6, H, W) float32, 归一化到 [0, 1]
#             combined = load_image_pair(rgb_path, thermal_path, size=(H, W))
#             batches.append(combined)
#         return np.stack(batches).astype(np.float32)  # (N, 6, H, W)
#
#     def get_batch_size(self):
#         return self.batch_size
#
#     def get_batch(self, names):
#         if self.current_index + self.batch_size > len(self.data):
#             return None
#
#         batch = self.data[self.current_index:self.current_index + self.batch_size]
#         self.current_index += self.batch_size
#
#         # Debug 检查
#         assert isinstance(batch, np.ndarray), "Batch is not ndarray"
#         assert batch.dtype == np.float32, f"Expected float32 but got {batch.dtype}"
#         assert batch.shape == (self.batch_size, *self.input_shape), f"Shape mismatch: {batch.shape}"
#
#         # 直接打平并转为 bytes
#         return [batch.tobytes()]
#
#     def read_calibration_cache(self):
#         if os.path.exists(self.cache_file):
#             print(f"Reading calibration cache from {self.cache_file}")
#             with open(self.cache_file, "rb") as f:
#                 return f.read()
#         return None
#
#     def write_calibration_cache(self, cache):
#         print(f"Writing calibration cache to {self.cache_file}")
#         with open(self.cache_file, "wb") as f:
#             f.write(cache)
import numpy as np
import tensorrt as trt

class DummyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_shape=(6, 224, 224), cache_file="calib_cache.bin", batch_size=1):
        super().__init__()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.current_index = 0
        self.data = np.zeros((10, *input_shape), dtype=np.float32)  # 构造 10 个 6x224x224 的 all-zero 数据

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.data):
            return None
        batch = self.data[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size

        assert isinstance(batch, np.ndarray)
        assert batch.dtype == np.float32
        return [batch.tobytes()]  # ✅ 返回列表里包一份 bytes

    def read_calibration_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                return f.read()
        except:
            return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
