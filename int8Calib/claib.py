import os
import numpy as np
# import pycuda
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import tensorrt as trt



class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, rgb_dir, thermal_dir, input_shape=(6, 224, 224), cache_file="calib_cache.bin", batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.input_shape = input_shape  # (C, H, W)
        self.cache_file = cache_file

        self.rgb_images = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))])
        self.thermal_images = sorted([os.path.join(thermal_dir, f) for f in os.listdir(thermal_dir) if f.endswith(('.jpg', '.png'))])
        assert len(self.rgb_images) == len(self.thermal_images), "RGB and thermal image counts do not match!"

        self.num_batches = len(self.rgb_images) // batch_size
        self.index = 0

        self.device_input = cuda.mem_alloc(
            batch_size * np.prod(input_shape).item() * np.dtype(np.float32).itemsize
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index >= self.num_batches:
            return None

        batch_data = []
        for i in range(self.batch_size):
            idx = self.index * self.batch_size + i

            rgb = cv2.imread(self.rgb_images[idx])
            thermal = cv2.imread(self.thermal_images[idx], cv2.IMREAD_GRAYSCALE)

            # Resize to target input shape
            h, w = self.input_shape[1], self.input_shape[2]
            rgb = cv2.resize(rgb, (w, h))
            thermal = cv2.resize(thermal, (w, h))

            # Normalize to [0, 1]
            rgb = rgb.astype(np.float32) / 255.0
            thermal = thermal.astype(np.float32) / 255.0

            # (3, H, W)
            rgb = rgb.transpose(2, 0, 1)
            thermal_stack = np.stack([thermal] * 3, axis=0)

            # (6, H, W)
            combined = np.concatenate([rgb, thermal_stack], axis=0)
            batch_data.append(combined)

        batch_np = np.ascontiguousarray(np.stack(batch_data))  # shape: (B, 6, H, W)
        cuda.memcpy_htod(self.device_input, batch_np)
        self.index += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)