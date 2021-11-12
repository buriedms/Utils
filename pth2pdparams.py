import torch
import paddle
import os

EXTENSIONS = ['pth',]


def is_file(filename):
    return any(filename.endswith(extension) for extension in EXTENSIONS)


def make_files(dir):
    files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_file(fname):
                path = os.path.join(root, fname)
                files.append(path)

    return files

def fix_path(path):
    return '/'.join(path.split('\\'))

def pth2pdparams(input_path, output_path):
    input_path = input_path
    output_path = output_path
    files_list=make_files(input_path)
    for i,file_path in enumerate(files_list):
        print('num:',i+1,'input_path:',file_path)
        # file_path=fix_path(file_path)
        state=paddle.load(file_path)
        os.remove(file_path)
        file_path=file_path.replace(input_path,output_path)
        file_path=file_path.replace('pth','pdparams')
        print('output_path:',file_path)
        paddle.save(state,file_path)

def pth2pdparams_rename(input_path, output_path):
    input_path = input_path
    output_path = output_path
    files_list = make_files(input_path)
    for i, file_path in enumerate(files_list):
        print('num:', i + 1, 'input_path:', file_path)
        origin_path=file_path
        # file_path = file_path.replace(input_path, output_path)
        file_path = file_path.replace('pth', 'pdparams')
        print('output_path:', file_path)
        os.rename(origin_path,file_path)


if __name__ == '__main__':
    # input_path=r'D:\Files\GitHub\Old2Life-Paddle\Global'
    # output_path=r'D:\Files\GitHub\Utils'
    # pth2pdparams_rename(input_path,output_path)
    test_path=r'D:\Files\GitHub\Utils\temp\test.pdparams'
    modelstate=paddle.load(r'D:\Files\GitHub\Utils\checkpoints\domainA_SR_old_photos\best_net_D.pdparams')
    state={'params':modelstate,'lr':0.5}
    paddle.save({'params':modelstate,'lr':0.5},test_path)
    try:
        state=paddle.load(test_path)
        print('import success')
        print(state.keys())
    except None as e:
        print(e)

