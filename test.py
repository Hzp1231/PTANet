# # 测试
# from ultralytics import YOLO
#
# if __name__ == '__main__':
#
#     # model = YOLO(r'E:\detect/baseline_Add_vtuav640/weights/best.pt')
#     model = YOLO(r'E:\TwoStream_Yolov8-main\yaml\SpaFre.yaml')
#     # metrics = model.val(data=r'E:\TwoStream_Yolov8-main/data/VTUAV_down.yaml', split='val', device=[0], imgsz=640, batch=1)
#     print(model)
# import torch
# import torch.nn as nn
#
# class ChannelAttentionModule(nn.Module):
#     def __init__(self, c1, reduction=16):
#         super(ChannelAttentionModule, self).__init__()
#         mid_channel = c1 // reduction
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.shared_MLP = nn.Sequential(
#             nn.Linear(in_features=c1, out_features=mid_channel),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Linear(in_features=mid_channel, out_features=c1)
#         )
#         self.act = nn.Sigmoid()
#         #self.act=nn.SiLU()
#     def forward(self, x):
#         avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
#         maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
#         return self.act(avgout + maxout)
#
# class FusionModule_ch(nn.Module):
#     def __init__(self, c):
#         super(FusionModule_ch, self).__init__()
#         # self.spatial_attention = SpatialAttentionModule()
#         self.sigle_channel_attention = ChannelAttentionModule(c)
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
#         self.act = nn.Sigmoid()
#         # self.double_channel_attention = ChannelAttentionModule(c2)
#
#     # v2
#     def forward(self, x):
#         # F_dif = x[0] - x[1]
#         # F_com = x[0] + x[1]
#         F_cat = torch.cat((x[1], x[2]), dim=1)
#         channelweight = self.sigle_channel_attention(F_cat).squeeze(-1).squeeze(-1)
#         # sigle_channelweight = self.sigle_channel_attention(x1)
#         # double_channelweight = self.double_channel_attention(x2)
#         top_values, indices = torch.topk(channelweight, k=int(F_cat.shape[1] / 2), dim=1)
#         # 调整 indices 形状，使其与 F_cat 匹配
#         # indices 原始形状: [batch_size, top_k]
#         indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, F_cat.shape[2], F_cat.shape[3])
#
#         # 使用 scatter 索引选取对应的特征图
#         selected_features = torch.gather(F_cat, dim=1, index=indices)
#
#         avgout = torch.mean(selected_features, dim=1, keepdim=True)
#         # # print(avgout.shape)
#         maxout, _ = torch.max(selected_features, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.act(self.conv2d(out))
#         out = x[0]*out
#
#         return out
#
# if __name__ == '__main__':
#
#     tensor1 = torch.randn(16, 64, 80, 80)
#     tensor2 = torch.randn(16, 64, 80, 80)
#     tensor3 = torch.randn(16, 64, 80, 80)
#     input = [tensor1, tensor2, tensor3]
#     model = FusionModule_ch(128)
#     out = model(input)
#     print(out.shape)

import matplotlib.pyplot as plt
import cv2

image = cv2.imread(r'B:\IGRASS\dfc25_track1_trainval\train\labels\TrainArea_379.tif', cv2.IMREAD_GRAYSCALE)



plt.imshow(image)

plt.show()