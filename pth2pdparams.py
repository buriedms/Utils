import torch
import paddle
import os
from model_paddle import DeeplabMulti,Res_Deeplab

EXTENSIONS = ['pth', ]


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
    files_list = make_files(input_path)
    for i, file_path in enumerate(files_list):
        print('num:', i + 1, 'input_path:', file_path)
        # file_path=fix_path(file_path)
        state = paddle.load(file_path)
        os.remove(file_path)
        file_path = file_path.replace(input_path, output_path)
        file_path = file_path.replace('pth', 'pdparams')
        print('output_path:', file_path)
        paddle.save(state, file_path)


def pth2pdparams_rename(input_path, output_path):
    input_path = input_path
    output_path = output_path
    files_list = make_files(input_path)
    for i, file_path in enumerate(files_list):
        print('num:', i + 1, 'input_path:', file_path)
        origin_path = file_path
        # file_path = file_path.replace(input_path, output_path)
        file_path = file_path.replace('pth', 'pdparams')
        print('output_path:', file_path)
        os.rename(origin_path, file_path)


def pth2pdparams_by_state(state_pth):
    state_pdparams = {}
    for key in state_pth:
        weight = state_pth[key].cpu().detach().numpy()
        if 'running_mean' in key:
            key = key.replace('running_mean', '_mean')
        if 'running_var' in key:
            key = key.replace('running_var', '_variance')
        # if 'classifier.weight' == key:
        #     weight = weight.transpose()
        state_pdparams[key] = weight
    return state_pdparams


def model_test(model, state):
    try:
        model.set_state_dict(state)
        print('set success !!!')
    except:
        raise ValueError('载入参数有误')


def pth2pdparams_fold(pth_fold_path, model=None):
    if not os.path.exists(pth_fold_path):
        raise ValueError(f'not exist {pth_fold_path} ~')
    for pth_path in os.listdir(pth_fold_path):
        print(f'Now we process {os.path.join(pth_fold_path, pth_path)}')
        state_pth_path = os.path.join(pth_fold_path, pth_path)
        pdparams_path = state_pth_path.replace('checkpoints', 'checkpoints_new')
        pdparams_path = pdparams_path.replace('pth', 'pdparams')
        # pdparams_path = pdparams_path.split('-')[0] + '.pdparams'

        state_pth = torch.load(state_pth_path, map_location='cpu')
        state_pth_list=list(state_pth.values())
        # print(state_pth_list[0].numpy()[0][0][0])
        # raise None
        new_params = model.state_dict().copy()
        for i in state_pth:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = state_pth[i]
                # print i_parts

        state_pdparams = pth2pdparams_by_state(new_params)

        model_test(model, state_pdparams)

        paddle.save(state_pdparams, pdparams_path)
        print(f'Successful conversion !!! || {pdparams_path}\n')


if __name__ == '__main__':
    state_pth_path = r'D:\Files\GitHub\Utils\checkpoints'
    model = DeeplabMulti(num_classes=19)
    # model = Res_Deeplab(num_classes=19)
    pth2pdparams_fold(state_pth_path, model=model)
