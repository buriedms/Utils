from tqdm import tqdm, trange
import os
import shutil

def copy_to(from_path,to_path):
    shutil.copy(from_path,to_path)


if __name__ == '__main__':
    txt_path=r'D:\Files\GitHub\AdvSemiSeg-Paddle\dataset\voc_list\val.txt'
    fold_path_from=r'E:\datasets\pascalvoc\VOCdevkit\VOC2012\JPEGImages'
    fold_path_to=r'D:\Files\GitHub\AdvSemiSeg-Paddle\dataset\VOC2012\JPEGImages'
    extend='.jpg'
    with open(txt_path,'r') as f:
        name_list=[name.strip()+extend if '.'not in name else name.strip() for name in f.readlines()]
    print('txt length :',len(name_list))
    for name in tqdm(name_list,desc='COPY IMG',unit='img'):
        src_name=os.path.join(fold_path_from,name)
        dst_name=os.path.join(fold_path_to,name)
        copy_to(src_name,dst_name)
    print('success copy ! ! !')