import os
import subprocess

ZIP_LIST = ['.zip']
EXT_LIST = ['.zip']
ZIP7_PATH = 'D:\\7zip\\7-Zip\\7z.exe'


def delete_old(path, ext_list=None):
    if not ext_list:
        ext_list = EXT_LIST
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] not in ext_list:
                continue
            src_path = os.path.join(root, file)
            os.remove(src_path)


def unzips(fold_path_from, fold_path_to, zip_list=None, zip7_path=ZIP7_PATH,new_flod=False):
    flag = False
    if not zip_list:
        zip_list = ZIP_LIST
    for root, dirs, files in os.walk(fold_path_from):
        # print('====',root,dirs)
        root_tar = root.replace(fold_path_from, fold_path_to)
        os.makedirs(root_tar, exist_ok=True)
        for file in files:
            # print(os.path.splitext(file))
            # print(os.path.join(root, file))
            if os.path.splitext(file)[1] not in zip_list:
                continue
            src_path = os.path.join(root, file)
            dst_path = os.path.join(root_tar, file.split('.')[0]) if new_flod else root_tar
            cmd = '\"{}\" x \"{}\" -o\"{}\" -r'.format(zip7_path, src_path, dst_path)
            print(src_path, '================>', dst_path)
            os.popen(cmd)
            flag = True
    # print('\n')
    print('Decompressing') if flag else print('Nothing')
    return flag


# def unzip_exe(fold_path_from, fold_path_to, zip_list=None, zip7_path=ZIP7_PATH):
#     flag = True
#     while (flag):
#         flag = unzips(fold_path_from, fold_path_to, zip_list, zip7_path)
#         fold_path_from = fold_path_to
#         print(flag)


if __name__ == '__main__':
    txt_path = r'D:\Files\GitHub\AdvSemiSeg-Paddle\dataset\voc_list\val.txt'
    fold_path_from = r'D:\Files\GitHub\AdaptSegNet-Paddle\data\GTA5'
    fold_path_to = r'D:\Files\GitHub\AdaptSegNet-Paddle\data\GTA5'
    # os.popen(f'rd/s/q \"{fold_path_to}\"')
    # unzips(fold_path_from, fold_path_to,new_flod=False)
    delete_old(fold_path_to)# 慎用 ！！！
