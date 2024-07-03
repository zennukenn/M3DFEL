import os
from PIL import Image
from multiprocessing import Pool, cpu_count


def convert_and_save_image(file_info):
    source_file_path, destination_file_path, quality = file_info
    try:
        # 创建目标文件夹（如果不存在）
        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)

        # 转换BMP为JPG并保存
        with Image.open(source_file_path) as bmp_image:
            bmp_image = bmp_image.convert("RGB")
            bmp_image.save(destination_file_path, "JPEG", quality=quality)

    except FileNotFoundError:
        print(f"文件未找到: {source_file_path}")
    except Exception as e:
        print(f"处理文件 {source_file_path} 时出错: {e}")


def process_directory(args):
    root, dir_name, source, destination, quality = args
    file_info_list = []

    dir_path = os.path.join(root, dir_name)
    bmp_files = [f for f in os.listdir(dir_path) if f.endswith('.bmp')]

    # 检查是否有140张以上的BMP图片
    if len(bmp_files) <= 140:
        print(f"{dir_path} 文件少于140张")

    for file in bmp_files:
        source_file_path = os.path.join(dir_path, file)

        # 构建目标文件路径
        relative_path = os.path.relpath(source_file_path, source)
        destination_file_path = os.path.join(destination, relative_path).replace(".bmp", ".jpg")

        # 添加到处理列表
        file_info_list.append((source_file_path, destination_file_path, quality))

    return file_info_list


def copy_and_convert_images(source, destination, quality=95):
    # 准备处理目录的参数
    args_list = []

    for root, dirs, files in os.walk(source):
        for dir_name in dirs:
            if "aligned" in dir_name:
                args_list.append((root, dir_name, source, destination, quality))

    # 使用多进程处理目录
    file_info_list = []
    with Pool(cpu_count()) as pool:
        results = pool.map(process_directory, args_list)
        for result in results:
            file_info_list.extend(result)

    # 使用多进程转换和保存图像
    with Pool(cpu_count()) as pool:
        pool.map(convert_and_save_image, file_info_list)


if __name__ == "__main__":
    # 定义源目录和目标目录
    source_dir = "D:\dataset\DAiSEE\DataSet\Train"
    destination_dir = "D:\dataset\DAiSEE\\aug\Train"

    # 执行复制和转换，指定JPEG质量为95（范围是1-100，100是最好的质量）
    copy_and_convert_images(source_dir, destination_dir, quality=100)
    print("完成！")
