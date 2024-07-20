import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
data_dir ='../datasets/DUTS_Test/result_8_base'
# 定义数据预处理和数据加载器
transform = transforms.Compose([transforms.ToTensor()])
dataset = ImageFolder(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 初始化一个字典用于存储每个类别的图像像素总和和计数
class_sums = {}
class_counts = {}

# 遍历数据集计算每个类别的图像像素总和和计数
for data, label in data_loader:
    data = data.squeeze(0)  # 去除批次维度 (1, C, H, W) -> (C, H, W)
    label = label.item()
    if label not in class_sums:
        class_sums[label] = torch.zeros_like(data)
        class_counts[label] = 0
    class_sums[label] += data
    class_counts[label] += 1

# 计算每个类别的均值原型
class_prototypes = {}
for label, sum_data in class_sums.items():
    mean_prototype = sum_data / class_counts[label]
    class_prototypes[label] = mean_prototype

# 保存每个类别的原型
save_dir = 'class_prototypes'
os.makedirs(save_dir, exist_ok=True)
for label, prototype in class_prototypes.items():
    torch.save(prototype, os.path.join(save_dir, f'prototype_{label}.pt'))
