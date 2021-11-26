from tqdm import tqdm, trange
import os
import shutil


def mkdir_tree(fold_path_from, fold_path_to):
    for root, dirs, files in os.walk(fold_path_from):
        root_tar = root.replace(fold_path_from, fold_path_to)
        os.makedirs(root_tar, exist_ok=True)


def copy_to(from_path, to_path, extend=None):
    if extend:
        from_path = from_path + '.' + extend
        to_path = to_path + '.' + extend
    shutil.copy(from_path, to_path)


def get_txt(txt_path, extend=None):
    with open(txt_path, 'r') as f:
        name_list = [name.strip() + extend if '.' not in name else name.strip() for name in f.readlines()]
    name_list = [name.split('/')[-1] for name in name_list]
    return name_list


def copy_by_text(fold_path_from, fold_path_to, txt_path, extend=None):
    mkdir_tree(fold_path_from, fold_path_to)
    name_list = get_txt(txt_path, extend=extend)
    for root, dirs, files in os.walk(fold_path_from):
        root_tar = root.replace(fold_path_from, fold_path_to)
        os.makedirs(root_tar, exist_ok=True)
        for file in files:
            if file not in name_list:
                continue
            os.makedirs(root_tar, exist_ok=True)
            src_path = os.path.join(root, file)
            dst_path = os.path.join(root_tar, file)
            copy_to(src_path, dst_path)
            print(f'success move {file}')


if __name__ == '__main__':
    fold_path_from = r'E:\datasets\Cityscapes\leftImg8bit_trainvaltest'
    fold_path_to = r'D:\Files\GitHub\AdaptSegNet-Paddle\data\Cityscapes\data'
    txt_list=[r'D:\Files\GitHub\AdaptSegNet-Paddle\dataset\cityscapes_list\val.txt',
              r'D:\Files\GitHub\AdaptSegNet-Paddle\dataset\cityscapes_list\train.txt',
              r'D:\Files\GitHub\AdaptSegNet-Paddle\dataset\cityscapes_list\label.txt']
    for txt_path in txt_list:
        copy_by_text(fold_path_from,fold_path_to,txt_path)
