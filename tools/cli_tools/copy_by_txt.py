"""
按txt内容复制文件工具

功能：
- 根据txt文件内容，从源目录复制指定文件到目标目录。
- 支持自动创建目标目录树。
- 支持文件扩展名补全。

用法示例：
    copy_by_text(fold_path_from, fold_path_to, txt_path, extend='.png')

依赖：os, shutil, tqdm
"""
from tqdm import tqdm
import os
import shutil

def mkdir_tree(fold_path_from, fold_path_to):
    """
    递归创建目标目录树
    :param fold_path_from: 源目录
    :param fold_path_to: 目标目录
    """
    for root, dirs, files in os.walk(fold_path_from):
        root_tar = root.replace(fold_path_from, fold_path_to)
        os.makedirs(root_tar, exist_ok=True)

def copy_to(from_path, to_path, extend=None):
    """
    复制单个文件
    :param from_path: 源文件路径
    :param to_path: 目标文件路径
    :param extend: 文件扩展名
    """
    if extend:
        from_path = from_path + '.' + extend
        to_path = to_path + '.' + extend
    shutil.copy(from_path, to_path)

def get_txt(txt_path, extend=None):
    """
    读取txt文件，获取文件名列表
    :param txt_path: txt文件路径
    :param extend: 文件扩展名
    :return: 文件名列表
    """
    with open(txt_path, 'r') as f:
        name_list = [name.strip() + extend if (extend and '.' not in name) else name.strip() for name in f.readlines()]
    name_list = [name.split('/')[-1] for name in name_list]
    return name_list

def copy_by_text(fold_path_from, fold_path_to, txt_path, extend=None):
    """
    按txt内容复制文件
    :param fold_path_from: 源目录
    :param fold_path_to: 目标目录
    :param txt_path: txt文件路径
    :param extend: 文件扩展名
    """
    mkdir_tree(fold_path_from, fold_path_to)
    name_list = get_txt(txt_path, extend=extend)
    for root, dirs, files in os.walk(fold_path_from):
        root_tar = root.replace(fold_path_from, fold_path_to)
        os.makedirs(root_tar, exist_ok=True)
        for file in files:
            if file not in name_list:
                continue
            src_path = os.path.join(root, file)
            dst_path = os.path.join(root_tar, file)
            copy_to(src_path, dst_path)
            print(f'success move {file}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='按txt内容复制文件工具')
    parser.add_argument('--src', required=True, help='源目录')
    parser.add_argument('--dst', required=True, help='目标目录')
    parser.add_argument('--txt', required=True, help='txt文件路径')
    parser.add_argument('--ext', default=None, help='文件扩展名（可选）')
    args = parser.parse_args()
    copy_by_text(args.src, args.dst, args.txt, extend=args.ext)
