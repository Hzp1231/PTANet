import tensorrt as trt

TRT_LOGGER = trt.Logger()
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
config.max_workspace_size = 1 << 30  # 1GB


from claib import EntropyCalibrator


calibrator = EntropyCalibrator(
    rgb_dir="/home/omnisky/hzp/vtuav_1.0_down/claib1/images/train",
    thermal_dir="/home/omnisky/hzp/vtuav_1.0_down/claib1/image/train",
    input_shape=(6, 640, 640),  # 根据你模型改
    cache_file="calib_cache.bin",
    batch_size=1
)

config.int8_calibrator = calibrator

engine = builder.build_engine(network, config)