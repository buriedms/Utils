"""
批量解压指定文件夹下的所有压缩包（命令行参数支持，自动查找7z路径）

功能：
- 支持批量解压指定目录下的所有zip压缩包。
- 默认自动查找环境变量中的7z.exe路径，更通用。
- 若未找到7z.exe，则要求用户通过参数指定路径。
- 可自定义源目录、目标目录、是否为每个压缩包新建文件夹。
- 支持命令行参数调用，无需修改代码。

用法示例：
    python unzips.py --src 源目录 --dst 目标目录 [--new_fold True]
    # 或直接import调用 unzips(...)

依赖：7-Zip（需安装并配置路径）、os、subprocess
"""
import os
import subprocess
import shutil
import sys

def find_7z_path():
    """
    自动查找7z可执行文件路径，兼容Windows和Linux
    :return: 7z可执行文件绝对路径或None
    """
    import shutil
    exe_name = "7z.exe" if os.name == "nt" else "7z"
    # 1. 先查环境变量
    exe_path = shutil.which(exe_name)
    if exe_path:
        return exe_path
    # 2. 常见默认安装路径
    if os.name == "nt":
        default_paths = [
            r"C:\\Program Files\\7-Zip\\7z.exe",
            r"C:\\Program Files (x86)\\7-Zip\\7z.exe",
            r"D:\software\7-Zip\\7z.exe"
        ]
    else:
        default_paths = ["/usr/bin/7z", "/usr/local/bin/7z"]
    for exe_path in default_paths:
        if os.path.isfile(exe_path):
            return exe_path
    return None

def delete_old(path, ext_list=None):
    """
    删除指定目录下所有指定扩展名的文件
    :param path: 目录路径
    :param ext_list: 扩展名列表
    """
    EXT_LIST = ['.zip']
    if not ext_list:
        ext_list = EXT_LIST
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] not in ext_list:
                continue
            src_path = os.path.join(root, file)
            os.remove(src_path)

def unzips(folder_path_from, folder_path_to, zip_list=None, zip7_path=None, new_flod=False):
    """
    批量解压指定目录下的所有压缩包
    :param folder_path_from: 源目录
    :param folder_path_to: 目标目录
    :param zip_list: 支持的压缩包类型
    :param zip7_path: 7z解压程序路径
    :param new_flod: 是否为每个压缩包新建文件夹
    :return: 是否有解压操作
    """
    ZIP_LIST = ['.zip']
    if not zip_list:
        zip_list = ZIP_LIST
    if not zip7_path:
        zip7_path = find_7z_path()
        if not zip7_path:
            print("未检测到7z/7z.exe，请通过 --zip7_path 参数指定7z解压程序路径！")
            zip7_path = input("请输入7z可执行文件的完整路径（如C:/Program Files/7-Zip/7z.exe或/usr/bin/7z）：").strip()
            if not os.path.isfile(zip7_path):
                print("输入的7z路径无效，程序终止。")
                sys.exit(1)
    flag = False
    for root, dirs, files in os.walk(folder_path_from):
        root_tar = root.replace(folder_path_from, folder_path_to)
        os.makedirs(root_tar, exist_ok=True)
        for file in files:
            if os.path.splitext(file)[1] not in zip_list:
                continue
            src_path = os.path.join(root, file)
            dst_path = os.path.join(root_tar, file.split('.')[0]) if new_flod else root_tar
            # 兼容Windows和Linux的命令
            cmd = f'"{zip7_path}" x "{src_path}" -o"{dst_path}" -r' if os.name == "nt" else f'{zip7_path} x "{src_path}" -o"{dst_path}" -y'
            print(f'{src_path} =================> {dst_path}')
            os.system(cmd)
            flag = True
    print('Decompressing' if flag else 'Nothing')
    return flag

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='批量解压指定文件夹下的所有压缩包')
    parser.add_argument('--src', required=True, help='源目录')
    parser.add_argument('--dst', required=True, help='目标目录')
    parser.add_argument('--new_fold', type=bool, default=False, help='是否为每个压缩包新建文件夹(True/False)')
    args = parser.parse_args()
    unzips(args.src, args.dst, new_flod=args.new_fold)
