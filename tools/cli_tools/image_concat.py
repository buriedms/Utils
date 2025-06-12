"""
图像拼接工具（基础库实现+外部接口重构）

功能：
- 支持将多个文件夹下的图片按顺序拼接为一张大图。
- 仅依赖PIL和numpy，无需paddle等深度学习库。
- 提供外部接口image_concat_main，便于包内调用。

用法示例：
    input_path_list = ["./folder1", "./folder2"]
    output_path = "./result"
    image_concat_main(input_path_list, output_path, save_name="result")

依赖：PIL, numpy, os
"""
import os
import numpy as np
from PIL import Image
from typing import List, Optional

def load_images_from_folder(folder: str, size: Optional[tuple]=None) -> List[Image.Image]:
    """
    加载文件夹下所有图片
    :param folder: 图片文件夹路径
    :param size: 统一resize尺寸 (w, h)，默认不变
    :return: PIL.Image列表
    """
    images = []
    for filename in sorted(os.listdir(folder)):
        file_path = os.path.join(folder, filename)
        try:
            img = Image.open(file_path).convert('RGB')
            if size:
                img = img.resize(size, Image.BILINEAR)
            images.append(img)
        except Exception as e:
            print(f"跳过无法打开的图片: {file_path}, 错误: {e}")
    return images

def concat_images_horizontally(images: List[Image.Image]) -> Image.Image:
    """
    横向拼接一组图片
    :param images: PIL.Image列表
    :return: 拼接后的大图
    """
    if not images:
        raise ValueError("图片列表为空")
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im

def concat_images_vertically(images: List[Image.Image]) -> Image.Image:
    """
    纵向拼接一组图片
    :param images: PIL.Image列表
    :return: 拼接后的大图
    """
    if not images:
        raise ValueError("图片列表为空")
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im

def image_concat_main(input_path_list: List[str], output_path: str, save_name: str = 'result', mode: str = 'horizontal', resize: Optional[tuple] = None):
    """
    外部接口：拼接多个文件夹下的图片
    :param input_path_list: 图片文件夹路径列表
    :param output_path: 输出文件夹
    :param save_name: 输出文件名
    :param mode: 拼接模式 'horizontal'或'vertical'
    :param resize: 统一resize尺寸 (w, h)，默认不变
    """
    # 每个文件夹一组，组内图片横向拼接，组间纵向拼接
    group_images = []
    for folder in input_path_list:
        imgs = load_images_from_folder(folder, size=resize)
        if not imgs:
            continue
        group_img = concat_images_horizontally(imgs) if mode == 'horizontal' else concat_images_vertically(imgs)
        group_images.append(group_img)
    if not group_images:
        raise ValueError("未找到可拼接的图片")
    # 多组图片纵向拼接
    final_img = concat_images_vertically(group_images) if mode == 'horizontal' else concat_images_horizontally(group_images)
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, save_name + '.png')
    final_img.save(save_path)
    print(f"拼接完成，保存路径: {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='图片拼接工具')
    parser.add_argument('--input', nargs='+', required=True, help='输入图片文件夹路径列表，空格分隔')
    parser.add_argument('--output', required=True, help='输出文件夹路径')
    parser.add_argument('--save_name', default='result', help='输出文件名（不带扩展名）')
    parser.add_argument('--mode', default='horizontal', choices=['horizontal', 'vertical'], help='拼接模式')
    parser.add_argument('--resize', type=int, nargs=2, default=None, metavar=('W', 'H'), help='统一resize尺寸，如 256 256')
    args = parser.parse_args()
    image_concat_main(
        input_path_list=args.input,
        output_path=args.output,
        save_name=args.save_name,
        mode=args.mode,
        resize=tuple(args.resize) if args.resize else None
    )
