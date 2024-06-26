import pickle

import torch
from pytorchvideo.models.hub import i3d_r50
#
# # 加载本地参数文件
# checkpoint = torch.load('I3D_8x8_R50.pyth')
#
# # 只提取模型的参数部分
# model_state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
#
# # 加载I3D模型架构，但不使用预训练参数
# model = i3d_r50(pretrained=False)
#
# # 将本地参数加载到模型中
# model.load_state_dict(model_state_dict, strict=False)
#
# # 打印模型结构以了解其层次
# print(model)











# import torch
# import torch.nn as nn
# import pickle
#
# # 定义扩展2D卷积核到3D卷积核的函数
# def expand_2d_to_3d(conv2d_weight, depth):
#     out_channels, in_channels, height, width = conv2d_weight.shape
#     conv3d_weight = conv2d_weight.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth
#     return conv3d_weight
#
# # 文件路径
# PATH_PRE_50 = 'resnet50_ft_weight.pkl'
#
# # 读取文件并转换为PyTorch张量
# with open(PATH_PRE_50, 'rb') as f:
#     obj = f.read()
# weights_2d = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
#
# # 创建一个新的字典来存储3D ResNet50的参数
# weights_3d = {}
#
# # 假设我们要扩展到深度为3的3D卷积核
# depth = 3
#
# # 遍历2D模型的参数并进行扩展
# for key, weight in weights_2d.items():
#     if 'conv' in key and len(weight.shape) == 4:
#         # 扩展2D卷积核到3D卷积核
#         weight_3d = expand_2d_to_3d(weight, depth)
#         weights_3d[key] = weight_3d
#         print(f"扩展卷积层 {key} 到 3D")
#     else:
#         # 非卷积层参数直接复制
#         weights_3d[key] = weight
#
# # 保存3D ResNet50的参数文件
# torch.save(weights_3d, 'resnet50_3d.pth')
#
# print('2D ResNet50参数文件已成功扩展到3D并保存为resnet50_3d.pth')
# import torch
# from torch import nn
# from pytorchvideo.models.hub import i3d_r50
#
# # 加载模型参数
# model_state_dict = torch.load('resnet50_3d.pth')
#
# # 加载I3D模型架构，但不使用预训练参数
# model = i3d_r50(pretrained=False)
# features = nn.Sequential(
#     *list(model.children())[:-1])
#
# # 将本地参数加载到模型中
# try:
#     model.load_state_dict(model_state_dict, strict=False)
#     print("模型参数加载成功")
# except RuntimeError as e:
#     print(f"模型参数加载失败: {e}")
#
# # 打印模型结构以了解其层次
# print(model)
#
# # 随机生成一个输入张量，检查模型是否能正常运行
# input_tensor = torch.randn(1, 3, 8, 224, 224)  # (batch_size, channels, num_frames, height, width)
#
# # 前向传播检查
# try:
#     output = features(input_tensor)
#     print("前向传播成功，输出形状:", output.shape)
# except Exception as e:
#     print(f"前向传播失败: {e}")
# import torch
# import numpy as np
# def inflate_conv2d_to_conv3d(weight2d, time_dim=3):
#     # 首先，确保权重是torch.Tensor
#     if isinstance(weight2d, np.ndarray):
#         weight2d = torch.from_numpy(weight2d)
#
#     # 重复卷积核权重以匹配时间维度
#     weight3d = weight2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
#     return weight3d / time_dim
#
# def convert_key_2d_to_3d(key2d):
#     key3d = key2d.replace('layer', 'blocks')
#     # key3d = key3d.replace('bn', 'norm')
#     # key3d = key3d.replace('conv', 'conv')
#     flag = True
#     if 'conv1' in key2d and 'blocks' in key3d:
#         key3d = key3d.replace('conv1', 'branch2.conv_a')
#     elif 'conv1' in key2d:
#         key3d = key3d.replace('conv1', 'blocks.0.conv')
#         flag = False
#     if 'bn1' in key2d and 'blocks' in key3d:
#         key3d = key3d.replace('bn1', 'branch2.norm_a')
#     elif 'bn1' in key2d:
#         key3d = key3d.replace('bn1', 'blocks.0.norm')
#         flag = False
#     if 'downsample.0' in key2d:
#         key3d = key3d.replace('downsample.0', 'branch1_conv')
#     if 'downsample.1' in key2d:
#         key3d = key3d.replace('downsample.1', 'branch1_norm')
#     if 'fc' in key2d:
#         key3d = key3d.replace('fc', 'head.proj')
#     if 'conv2' in key2d:
#         key3d = key3d.replace('conv2', 'branch2.conv_b')
#     if 'conv3' in key2d:
#         key3d = key3d.replace('conv3', 'branch2.conv_c')
#     if 'bn2' in key2d:
#         key3d = key3d.replace('bn2', 'branch2.norm_b')
#     if 'bn3' in key2d:
#         key3d = key3d.replace('bn3', 'branch2.norm_c')
#
#
#     # 插入 res_blocks
#     if flag:
#         parts = key3d.split('.')
#         # 找到 blocks 后面的数字部分
#         for i, part in enumerate(parts):
#             if part.startswith('blocks'):
#                 num_part = part[len('blocks'):]
#                 parts[i] = 'blocks'
#                 parts.insert(i + 1, num_part)
#                 parts.insert(i + 2, 'res_blocks')
#                 break
#         key3d = '.'.join(parts)
#
#     return key3d
#
# # 假设你已经有一个加载了2D权重的字典
#
# def load_model_weights(file_path):
#     # 使用 'rb' 模式打开文件，'rb' 代表 'read binary'
#     with open(file_path, 'rb') as file:
#         # 使用 pickle 加载模型权重
#         model_weights = pickle.load(file)
#     return model_weights
#
# # 假设你的文件名是 'resnet50_3d_weights.pkl'
# weights_path = 'resnet50_ft_weight.pkl'
# model_weights = load_model_weights(weights_path)
# # 创建3D模型权重的字典
# model_3d_weights = {}
# i = 0
# # 对每个键进行处理和膨胀
# for key2d, weight2d in model_weights.items():
#     key3d = convert_key_2d_to_3d(key2d)
#
#     if 'conv' in key2d:
#         # 假设每个卷积层的时间维度是3
#         i=i+1
#         model_3d_weights[key3d] = inflate_conv2d_to_conv3d(weight2d, time_dim=3)
#         print(f'{i}')
#     else:
#         # 对于非卷积层参数，直接赋值
#         model_3d_weights[key3d] = weight2d
#         print(f'22')
#
# # 使用torch.save保存膨胀后的3D模型权重
# # print(model_3d_weights)
# # torch.save(model_3d_weights, 'resnet50_3d_weights.pth')

import torch
import pickle
import numpy as np
from pytorchvideo.models.resnet import create_resnet
import torch.nn as nn


def inflate_conv2d_to_conv3d(weight2d, time_dim=3):
    if isinstance(weight2d, np.ndarray):
        weight2d = torch.from_numpy(weight2d)
    weight3d = weight2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)

    return weight3d / time_dim


def convert_key_2d_to_3d(key2d):
    key3d = key2d.replace('layer', 'blocks')
    # key3d = key3d.replace('bn', 'norm')
    # key3d = key3d.replace('conv', 'conv')
    flag = True
    if 'conv1' in key2d and 'blocks' in key3d:
        key3d = key3d.replace('conv1', 'branch2.conv_a')
    elif 'conv1' in key2d:
        key3d = key3d.replace('conv1', 'blocks.0.conv')
        flag = False
    if 'bn1' in key2d and 'blocks' in key3d:
        key3d = key3d.replace('bn1', 'branch2.norm_a')
    elif 'bn1' in key2d:
        key3d = key3d.replace('bn1', 'blocks.0.norm')
        flag = False
    if 'downsample.0' in key2d:
        key3d = key3d.replace('downsample.0', 'branch1_conv')
    if 'downsample.1' in key2d:
        key3d = key3d.replace('downsample.1', 'branch1_norm')
    if 'fc' in key2d:
        key3d = key3d.replace('fc', 'head.proj')
    if 'conv2' in key2d:
        key3d = key3d.replace('conv2', 'branch2.conv_b')
    if 'conv3' in key2d:
        key3d = key3d.replace('conv3', 'branch2.conv_c')
    if 'bn2' in key2d:
        key3d = key3d.replace('bn2', 'branch2.norm_b')
    if 'bn3' in key2d:
        key3d = key3d.replace('bn3', 'branch2.norm_c')


    # 插入 res_blocks
    if flag:
        parts = key3d.split('.')
        # 找到 blocks 后面的数字部分
        for i, part in enumerate(parts):
            if part.startswith('blocks'):
                num_part = part[len('blocks'):]
                parts[i] = 'blocks'
                parts.insert(i + 1, num_part)
                parts.insert(i + 2, 'res_blocks')
                break
        key3d = '.'.join(parts)

    return key3d


def load_model_weights(file_path):
    try:
        with open(file_path, 'rb') as file:
            model_weights = pickle.load(file)
        return model_weights
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return {}


def process_and_save_weights(weights_path, save_path):
    model_weights = load_model_weights(weights_path)
    model_3d_weights = {}

    # 定义每个卷积层的时间维度
    conv_time_dims = {
        'blocks.0.conv': 3,
        'blocks.1.res_blocks.1.branch2.conv_a': 1,
        'blocks.1.res_blocks.2.branch2.conv_a': 1,
        'blocks.2.res_blocks.1.branch2.conv_a': 1,
        'blocks.2.res_blocks.2.branch2.conv_a': 1,
        'blocks.2.res_blocks.3.branch2.conv_a': 1,
        'blocks.1.res_blocks.0.branch2.conv_a': 1,
        'blocks.2.res_blocks.0.branch2.conv_a': 1,
        'blocks.3.res_blocks.0.branch2.conv_a': 3,
        'blocks.3.res_blocks.1.branch2.conv_a': 3,
        'blocks.3.res_blocks.2.branch2.conv_a': 3,
        'blocks.3.res_blocks.3.branch2.conv_a': 3,
        'blocks.3.res_blocks.4.branch2.conv_a': 3,
        'blocks.3.res_blocks.5.branch2.conv_a': 3,
        'blocks.4.res_blocks.0.branch2.conv_a': 3,
        'blocks.4.res_blocks.1.branch2.conv_a': 3,
        'blocks.4.res_blocks.2.branch2.conv_a': 3,
        'branch2.conv_b': 1,
        'branch2.conv_c': 1,
        'branch1_conv': 1
    }
    i = 0
    for key2d, weight2d in model_weights.items():
        key3d = convert_key_2d_to_3d(key2d)
        if 'conv' in key3d and len(weight2d.shape) == 4:  # Only inflate conv layers with 4D weight
            # 查找对应层的时间维度
            i =i+1

            time_dim = next((dim for name, dim in conv_time_dims.items() if name in key3d), 9)
            model_3d_weights[key3d] = inflate_conv2d_to_conv3d(weight2d, time_dim=time_dim)
            print(i, time_dim)
        else:
            # 确保所有权重参数都是torch.Tensor
            if isinstance(weight2d, np.ndarray):
                weight2d = torch.from_numpy(weight2d)
            model_3d_weights[key3d] = weight2d


    torch.save(model_3d_weights, save_path)
    print(f"3D weights saved to {save_path}")


# # 创建 3D ResNet-50 模型
# resnet3d = create_resnet(
#     input_channel=3,
#     model_depth=50,
#     norm=nn.BatchNorm3d,  # 使用 BatchNorm3d
#     activation=nn.ReLU  # 使用 ReLU 激活函数
# )
resnet3d = i3d_r50(pretrained=False)
# 打印模型结构
print(resnet3d)
#
# # 使用函数处理权重和保存
process_and_save_weights('resnet50_ft_weight.pkl', 'resnet50_3d_weights11.pth')
#
# # 加载保存的3D模型权重到模型中
model_3d_weights = torch.load('resnet50_3d_weights11.pth')
resnet3d.load_state_dict(model_3d_weights,strict=False)
print("Model loaded with 3D weights.")
