import os
import pandas as pd
import imageio.v3 as iio
import imgaug.augmenters as iaa
from multiprocessing import Pool

# 定义数据增强序列
augmenters = [
    None,  # part1 不做任何数据增强
    iaa.Sequential([
        iaa.Fliplr(1),  # 100%的概率水平翻转
        iaa.Multiply((0.7, 1.5)),  # 随机调整亮度
    ]),
    iaa.Sequential([
        iaa.Crop(percent=(0, 0.1)),  # 随机裁剪
        iaa.Grayscale(alpha=(0.0, 1.0)),  # 随机灰度化
    ]),
    iaa.Sequential([
        iaa.Affine(
            rotate=(-60, 60),  # 随机旋转角度在 -60 到 60 之间
            # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # 随机平移
            scale=(0.8, 1.0),  # 随机缩放
        ),
        iaa.LinearContrast((0.5,1.2)),# 对比度归一化
    ]),
    iaa.Sequential([
        iaa.AddToHueAndSaturation((-30, 30)),  # 随机调整色调和饱和度
        iaa.AdditiveGaussianNoise(scale=(0, 0.02 * 255)),  # 添加高斯噪声
    ])
]

# 数据增强处理函数
def process_clip(clip_id):
    try:
        # 确定数据增强器
        part_idx = int(clip_id.split('_part')[1]) - 1 # 假设part格式为 'ID_partx'
        augmenter = augmenters[part_idx]

        # 生成随机参数以确保同一视频组使用相同的数据增强参数（如果需要）
        augmenter_det = augmenter.to_deterministic() if augmenter else None

        part_str = clip_id.split('_')[-1]
        clipid_str = clip_id.split('_')[0]
        frames_path = os.path.join(dataset_dir, part_str, clipid_str[:6], clip_id)
        frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg')])
        if not len(frame_files) == 60:
            print(f"{frames_path},有问题")
        # 遍历每个视频帧图像
        for frame_file in frame_files:
            frame_path = os.path.join(frames_path, frame_file)
            try:

                # 对图像进行数据增强（如果需要）
                if augmenter_det:
                    # 读取图像
                    image = iio.imread(frame_path)
                    augmented_image = augmenter_det(image=image)
                    # 保存增强后的图像（覆盖原图像）
                    iio.imwrite(frame_path, augmented_image,quality=90)
                else:
                    # 不做增强，直接保存原图像
                    pass
            except Exception as e:
                print(f"Error processing frame {frame_file} in {clip_id}: {e}")
    except Exception as e:
        print(f"Error processing clip {clip_id}: {e}")

# 读取CSV文件
csv_file = '../../data/Daisee_copy5/Labels/TestLabels.csv'  # 请替换为实际的CSV文件路径
df = pd.read_csv(csv_file)

dataset_dir = '../../data/Daisee_copy5/DataSet/Test'

# 准备多进程任务
tasks = [clip_id for clip_id in df['ClipID'].unique()]

# 使用多进程处理
if __name__ == '__main__':
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_clip, tasks)

print("数据增强完成！")
