import os
import pandas as pd
import shutil
from concurrent.futures import ThreadPoolExecutor



# 函数：提取每份的帧
def extract_partitioned_frames(video_path, part_index, new_clip_path,total_parts=5, frames_per_part=60):
    frames = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
    total_frames = len(frames)
    selected_frames = []
    start_frame = part_index
    last_frame = None

    for i in range(frames_per_part):
        frame_index = start_frame + i * total_parts
        if frame_index < total_frames:
            selected_frames.append(frames[frame_index])

            last_frame = frames[frame_index]
        elif last_frame:
            # 复制最后一帧并添加到selected_frames中
            new_frame = f"{last_frame.split('.')[0]}_{i}.jpg"
            if not os.path.exists(new_clip_path):
                os.makedirs(new_clip_path)
            shutil.copy(os.path.join(video_path, last_frame), os.path.join(new_clip_path, new_frame))

        else:
            print(f"{video_path}有问题")

    return selected_frames

# 函数：处理每个视频文件夹
def process_clip(index, row):
    clip_id = row['ClipID'].split(".")[0]
    video_path = os.path.join(data_path, f'{clip_id[:6]}/{clip_id}/output/{clip_id}_aligned')
    new_rows = []
    if os.path.exists(video_path):
        frames = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        total_frames = len(frames)
        if total_frames < 300:
            print(video_path)

        for part_index in range(5):

            new_clip_id = f'{clip_id}_part{part_index + 1}'
            new_clip_path = os.path.join(output_path,f"part{part_index + 1}",f"{clip_id[:6]}", new_clip_id)
            selected_frames = extract_partitioned_frames(video_path, part_index,new_clip_path)
            if not selected_frames:
                print(f"{video_path} 无图片")
                continue


            if not os.path.exists(new_clip_path):
                os.makedirs(new_clip_path)

            # 保存选取的帧到新的文件夹
            for frame in selected_frames:
                src_frame_path = os.path.join(video_path, frame)
                dst_frame_path = os.path.join(new_clip_path, frame)
                shutil.copyfile(src_frame_path, dst_frame_path)

            # 更新标签数据
            new_row = {
                'ClipID': new_clip_id,
                'Boredom': row['Boredom'],
                'Engagement': row['Engagement'],
                'Confusion': row['Confusion'],
                'Frustration': row['Frustration ']
            }
            new_rows.append(new_row)
    else:
        print(f"{video_path} not exist")

    return new_rows


# 设置数据集路径和标签文件路径
data_path = '../../data/Daisee_aug/DataSet/Test'
label_file_path = '../../data/Daisee_aug/Labels/TestLabels.csv'
output_path = '../../data/Daisee_copy5/DataSet/Test'  # 保存新帧的路径

# 创建输出目录
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 读取标签文件
labels = pd.read_csv(label_file_path)
# 使用线程池并行处理
new_data = []
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_clip, index, row) for index, row in labels.iterrows()]
    for future in futures:
        result = future.result()
        if result:
            new_data.extend(result)

# 更新标签文件
new_labels = pd.DataFrame(new_data)
new_labels.to_csv('../../data/Daisee_copy5/Labels/TestLabels.csv', index=False)

print("新的标签文件已保存到 /Daisee_copy5/Labels/TestLabels.csv")