import numpy as np

# import torch
# import torch.nn as nn
# from torch.utils import data, model_zoo
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F

import mindspore
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)

import paddle
# import paddle.nn as nn

from src.models.mindspore import get_deeplab_v2
from src.models.paddlepaddle import DeeplabMulti

# from paddlepaddle import DeeplabMulti
# import torch
# from utils.loss import cross_entropy2d

"""
paddle model to torch model
"""


def update_name(list_old):
    list = ['.'.join(name.split('.')[0:]) for name in list_old]
    list_ = []
    for name, name_old in zip(list, list_old):
        pre_name = name.split('.')[0]
        # if pre_name != 'layer5' and pre_name != 'layer6':
        #     list_.append((name, name_old))
        list_.append((name, name_old))
    return list_


def name_raplace(name):
    # torch 和mindspore部分参数名不一致，进行修改，以下修改bn层参数名
    # bn_dict = {'weight': 'gamma', 'bias': 'beta',
    #            'running_mean': 'moving_mean', 'running_variance': 'moving_variance'}
    bn_dict = {'weight': 'gamma', 'bias': 'beta',
               '_mean': 'moving_mean', '_variance': 'moving_variance'}
    old = name.split('.')[-1]
    if 'bn' in name:
        return name.replace(old, bn_dict[old])
    elif ('downsample' in name) and (name.split('.')[-2] != '0'):
        # print(name)
        return name.replace(old, bn_dict[old])
    else:

        return name


def updata_torch_to_ms(static_dict_ms, static_dict_torch):
    # new_static_dict=dict()
    key_list = update_name(static_dict_torch.keys())  # 我的预训练模型需要舍去resnet后两层，所以有这个步骤
    for key, key_old in key_list:
        key_ = name_raplace(key)  # 置换参数名
        param = static_dict_torch[key_old]
        new_param = mindspore.Tensor(param.numpy())
        # print(key, '\t||||\t', key_)
        static_dict_ms[key_].set_data(new_param)  # 加载新权重


static_dict1 = paddle.load(r"D:\Files\GitHub\Utils\model_paddle\GTA5_Best.pdparams")
# print(saved_static_dict)
model = get_deeplab_v2()
model_p = DeeplabMulti()

print('原始参数：', list(model.parameters_dict().values())[0][0][0][0])

static_dict2 = model.parameters_dict()
keys_torch = static_dict1.keys()
keys_ms = static_dict2.keys()
print(keys_torch)
print(keys_ms)
updata_torch_to_ms(static_dict2, static_dict1)
value_list1 = list(static_dict1.values())
value_list2 = list(static_dict2.values())
print('目标参数：', value_list1[0].numpy()[0][0][0])
print('转换参数：', value_list2[0].asnumpy()[0][0][0])
mindspore.load_param_into_net(model, static_dict2)
model_p.load_dict(static_dict1)
# print(model_p.parameters())
# raise None


np.random.seed(100)
data = np.random.randn(1, 3, 128, 128)

data_ms = mindspore.Tensor(data, mindspore.float32)
data_p = paddle.to_tensor(data, paddle.float32)
print(data_ms.shape, data_p.shape)
model.set_train(False)
model_p.eval()
out1_ms, out2_ms = model(data_ms)
out1_p, out2_p = model_p(data_p)
print(out2_ms.shape, out2_ms[0][0][0])
print(out2_p.shape, out2_p[0][0][0])
# print('保存参数：', list(model.parameters_dict().values())[0][0][0][0])
# parameters_ms = [{'name': name, 'data': data} for name, data in model.parameters_and_names(name_prefix='net_G')]
#
# mindspore.save_checkpoint(parameters_ms, r'mindspore/ResNet_best_DeepLab_2.ckpt')
#
# state_dict = mindspore.load_checkpoint(r'mindspore/ResNet_best_DeepLab.ckpt')
# print(state_dict.keys())
# for key_old, key in zip(keys_ms, state_dict.keys()):
#     if 'net_G.' + key_old != key:
#         print(f'{key_old}\t|||{key}')
