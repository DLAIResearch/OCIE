import os
import torch

# 模型文件夹的路径，假设所有的模型文件都在同一文件夹下
model_folder_path = "class_prototypes"

# 创建一个空的字典，用于存储多个模型
models_dict = {}

# 遍历模型文件夹
for filename in os.listdir(model_folder_path):
    if filename.endswith(".pt"):
        model_path = os.path.join(model_folder_path, filename)
        # 从文件名中提取模型的名称
        # model_name = os.path.splitext(filename)[0]
        model_name=''.join([char for char in filename if char.isnumeric()])
        # 加载模型文件
        model = torch.load(model_path)

        # 将模型存储在字典中，以模型名称为键
        models_dict[int(model_name)] = model

a=models_dict[3]
# 保存多个模型到一个文件
torch.save(models_dict, "models_archive_imageNet_100.pt")