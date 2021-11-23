import numpy as np
import os
import shutil

def copy_to(from_path,to_path):
    shutil.copy(from_path,to_path)


if __name__ == '__main__':
    txt_path=r'D:\Files\GitHub\AdvSemiSeg-Paddle\dataset\voc_list\val.txt'
    fold_path_from=r'D:\Files\GitHub\AdvSemiSeg-Paddle\dataset\VOC2012\JPEGImages'
    fold_path_to=r'D:\Files\GitHub\Utils\temp'
    extend='.jpg'
    with open(txt_path,'r') as f:
        name_list=[name.strip()+extend for name in f.readlines()]
    print(name_list)
    print('txt length :',len(name_list))
    for name in name_list:
        src_name=os.path.join(fold_path_from,name)
        dst_name=os.path.join(fold_path_to,name)
        copy_to(src_name,dst_name)