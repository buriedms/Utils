import argparse
import numpy as np

# import torch
# import torch.nn as nn
# from torch.utils import data, model_zoo
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

import sys
import os
import os.path as osp
# import matplotlib.pyplot as plt
import random

from model_mindspore.deeplab_multi import DeeplabMulti
import torch


### 加载划分函数

def split_checkpoint(checkpoint, split_list=None):
    """
    Input：
    checkpoint:待划分的模型参数
    split_list:待划分的模型参数前缀名
    """
    if split_list == None:
        return checkpoint
    checkpoint_dict = {name: {} for name in split_list}
    for key, value in checkpoint.items():
        prefix = key.split('.')[0]
        if prefix not in checkpoint_dict:
            checkpoint_dict[key] = value.asnumpy()
            continue
        name = key.replace(prefix + '.', '')
        checkpoint_dict[prefix][name] = value
    return checkpoint_dict


def model_save_multi(save_path: str,
                     models_dict: dict,
                     append_dict=None, print_save=True) -> None:
    """
    Input:
    save_path:模型保存的路径
    models_dict:多模型字典，例如：{'net_G': model,'net_D1': model_D1,'net_D2': model_D2}
    append_dict:附加信息字典，例如：{'iter': 10, 'lr': 0.01, 'Acc': 0.98}
    print_save:是否打印模型保存路径
    Output：
    None
    """
    params_list = []
    for model_name, model in models_dict.items():
        for name, param in model.parameters_and_names(name_prefix=model_name):
            params_list.append({'name': name, 'data': param})
    mindspore.save_checkpoint(params_list, save_path, append_dict=append_dict)
    if print_save:
        print('Save success , The model save path : {}'.format(save_path))


# 置换参数名前缀
def change_param_name(name: str, name_change=None):
    if name_change is None:
        name_change = {}
    name_change_ = name_change
    # if prefix != '':
    #     name = name.replace(prefix + '.', '')
    # if prefix_new != '':
    #     name = prefix_new + '.' + name
    if '' in name_change:
        name = name_change[''] + '.' + name
        del name_change['']
    # name = name.replace(name_change_)

    for key in name_change:
        if key in name:
            name = name.replace(key, name_change_[key])
            if name[0] == '.':
                name = name[1:]
    # if 'patch' in name:
    #     print('\ninsert:', name)
    #     raise None
    return name


# 置换参数名 根据自己的需求来自定义设置
def name_raplace(name):
    # torch 和mindspore部分参数名不一致，进行修改，以下修改bn层参数名
    norm_dict = {'weight': 'gamma', 'bias': 'beta',
                 'running_mean': 'moving_mean', 'running_var': 'moving_variance'}  # bn层的映射关系表
    # transition = {'weight': 'gamma', 'bias': 'beta', 'running_mean': 'moving_mean', 'running_var': 'moving_variance'}
    old = name.split('.')[-1]
    flag = name.split('.')[-2]
    # return name

    # if (flag != '0') and ('conv' not in flag) and ('classifier' not in flag):
    #     return name.replace(old,norm_dict[old])
    # if name not in norm_dict:
    #     return name

    if 'norm' in name or 'bn' in name:
        # print(f'old name:\t{name}')
        try:
            return name.replace(old, norm_dict[old])
        except:
            print('skip param : {}'.format(name))
            return name
    elif ('downsample' in name) and (name.split('.')[-2] != '0'):
        #     # print(name)
        return name.replace(old, norm_dict[old])
    elif ('fuse_layers' in name) and (name.split('.')[-2] != '0'):
        #     # print(name)
        return name.replace(old, norm_dict[old])
    elif ('transition' in name) and (name.split('.')[-2] != '0'):
        #     # print(name)
        return name.replace(old, norm_dict[old])
    else:
        return name


# 更换前缀，置换参数名，并且实现删除其中特定层
def update_name(list_old, name_change=None, filter_list=None):
    if name_change is None:
        name_change = {}
    list_new = [change_param_name(name, name_change=name_change) for name in list_old]
    # print(list_)
    # raise None
    list_ = []
    if not filter_list:
        filter_list = []
    for name, name_old in zip(list_new, list_old):
        skip_flag = False
        for layer in filter_list:
            if layer in name:
                skip_flag = True
                list_.append((name_old, 'delete'))
                break
        if not skip_flag:
            list_.append((name_raplace(name), name_old))
    return list_


# 在原地址位置生成原名称的ckpt权重文件，并且生成原名称+'-pth-2-ckpt'的txt权重对应文件。
def update_torch_to_ms(pth_saved_path, ms_save_path='', txt_save_path='', space=80, name_change=None, filter_list=None):
    """
    :param pth_saved_path: pth load path
    :param ms_save_path:  ckpt save path
    :param txt_save_path: pth 2 ckpt log save path
    :param space: control the space in log file
    :param name_change: the parameter's name change rule: pth to ckpt ; type: dict{pth_name:ckpt_name}
    :param filter_list: The filter list of parameters : list ['pth_name1','pth_name2']
    :return: the changed ckpt
    """
    if not ms_save_path:
        ms_save_path = pth_saved_path.replace('.pth', '.ckpt')
    if not txt_save_path:
        txt_save_path = pth_saved_path.replace('.pth', '-pth-2-ckpt.txt')
    logger = open(txt_save_path, 'w')
    key, key_ = 'pth parameters', 'ckpt parameters'
    logger.write(key + '-' * (space - len(key)) + key_ + '\n' + '\n')
    static_dict_pth = torch.load(pth_saved_path, map_location='cpu')
    print("==" * 40, '\n', "pytorch parameters:")
    if not name_change:
        name_change = {}
    for key in static_dict_pth.keys():
        print(key)
    print("==" * 40)
    # new_static_dict=dict()
    key_list = update_name(static_dict_pth.keys(), name_change=name_change, filter_list=filter_list)
    static_list_ms = list()

    logger.write("The Translation log\n"
                 "The Number of Pytorch Parameters : \t{0:}\n"
                 "The Number of Mindspore Parameters : {1:}\n"
                 "The Number of Filtered Parameters : \t{2:}\n"
                 "The Flag of Filtered List : \t{3:}\n"
                 "The Old Param Name : \t{4:}\n"
                 "The New Param Name : \t {5:}\n"
                 "\n\n".format(
        len(static_dict_pth), len(list(filter(lambda x: x[1] != 'delete', key_list))), len(static_dict_pth) - len(list(filter(lambda x: x[1] != 'delete', key_list))), filter_list, list(name_change.keys()), list(name_change.values())
    )
    )

    logger.write('The Changed Parameter Table:\n')
    logger.write('The Changed Number:\t{}\n'.format(sum([int(key != key_old) for key, key_old in key_list])))
    for key, key_old in key_list:
        if key != key_old:
            logger.write(key_old + '-' * (space - len(key_old)) + key + '\n')
    logger.write('\n\n')

    logger.write('The All Parameter Table:\n')
    for key, key_old in key_list:
        # 一对一置换自定义
        # if key == 'norm.beta' or key == 'norm.gamma':
        #     # key = '3.'.join(key.split('.'))
        #     key = key.replace('norm', '3')
        # print(key)
        # raise None
        if key_old == 'delete':
            continue
        param = static_dict_pth[key_old]
        new_param = mindspore.Parameter(param.detach().numpy())
        print(key_old, '\t||||\t', key)
        logger.write(key_old + '-' * (space - len(key_old)) + key + '\n')
        static_list_ms.append({'name': key, 'data': new_param})  # 加载新权重
    logger.write('\n\n')

    logger.write('The Original Pytorch Parameters:\n')
    for key, value in static_dict_pth.items():
        logger.write(f"{key}\t{value.shape}\n")
    logger.write('\n\n')

    logger.write('The Mindspore Parameters:\n')
    for param in static_list_ms:
        logger.write(f"{param['name']}\t{param['data']}\n")
    logger.write('\n\n')

    print(static_list_ms)
    mindspore.save_checkpoint(static_list_ms, ms_save_path)
    logger.close()
    return {data['name']: data['data'] for data in static_list_ms}


if __name__ == '__main__':
    # pth_saved_path = r"D:\Files\GitHub\Utils\temp\checkpoints\hrnetv2_w48_imagenet_pretrained.pth"
    # pth_saved_path = r"D:\wendang\Github\seaelm\Utils\temp\model_pth\GTA5_init.pth"
    pth_saved_path = r"D:\wendang\Github\seaelm\Utils\temp\model_pth\SYNTHIA_init.pth"
    name_change = {'projection': 'proj',
                   'stages': '',
                   'attn.w_msa': 'attn',
                   'relative_position_bias_table': 'relative_bias.relative_position_bias_table',
                   'relative_position_index': 'relative_bias.index',
                   'ffn.layers.0.0': 'mlp.fc1',
                   'ffn.layers.1': 'mlp.fc2',
                   '1.blocks': '0.blocks.1.blocks',
                   '2.blocks': '0.blocks.2.blocks',
                   '3.blocks': '0.blocks.3.blocks',
                   # '0.downsample':'0.blocks.0.downsample',
                   '1.downsample': '0.blocks.1.downsample',
                   '2.downsample': '0.blocks.2.downsample', }
    # name_change = {}
    # filter_list = ['layer5', 'layer6']
    # filter_list = ['num_batches_tracked', 'classifier']
    # filter_list = ['mask']
    # filter_list = ['num_batches_tracked', 'incre_modules', 'downsamp_modules', 'final_layer', 'classifier']
    filter_list = ['num_batches_tracked']

    # static_dict_ms = update_torch_to_ms(pth_saved_path, name_change=name_change, filter_list=filter_list)
    static_dict_ms = update_torch_to_ms(pth_saved_path, filter_list=filter_list)
