#训练
import torch

from ultralytics import YOLO
from thop import profile
import ultralytics.nn.tasks
# model = YOLO(r'D:\桌面\行人车辆论文\onnx\mfdet/best.pt')
# model = YOLO('/home/omnisky/runs/detect/yolov8n_pc2f_mpf_train/weights/best.pt')
# input_tensor = torch.randn(16, 6, 640, 640)
# model = YOLO('E:\TwoStream_Yolov8-main\yaml\ADDyolov8.yaml')
# total_params = sum(p.numel() for p in model.parameters())
# print(f'Total parameters: {total_params}')
# flops, _ = profile(model, inputs=(input_tensor,))

# results = model.train(data='E:\TwoStream_Yolov8-main\data\VTUAV_down.yaml', batch=1, device=[0], epochs=400)
model.val(data='/home/omnisky/hzp/TwoStream_Yolov8-main/data/VTMOT.yaml',batch=16,device=[0],split='test')
# print(model)
