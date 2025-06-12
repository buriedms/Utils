"""
paddle模型参数转MindSpore模型参数工具

功能：
- paddle参数结构与MindSpore参数结构的自动映射与转换。
- 适用于迁移学习、模型兼容、参数对齐等。

依赖：mindspore, paddle, numpy
"""
import numpy as np
import mindspore
from mindspore import context
import paddle

context.set_context(mode=context.GRAPH_MODE)

def update_name(list_old):
    """生成参数名映射表"""
    return [('.'.join(name.split('.')), name) for name in list_old]

def name_raplace(name):
    """BN层参数名映射"""
    bn_dict = {'weight': 'gamma', 'bias': 'beta',
               '_mean': 'moving_mean', '_variance': 'moving_variance'}
    old = name.split('.')[-1]
    if 'bn' in name:
        return name.replace(old, bn_dict.get(old, old))
    elif ('downsample' in name) and (name.split('.')[-2] != '0'):
        return name.replace(old, bn_dict.get(old, old))
    else:
        return name

def update_paddle_to_ms(ms_model, paddle_state_dict):
    """将paddle参数加载到mindspore模型参数字典"""
    key_list = update_name(paddle_state_dict.keys())
    ms_state_dict = ms_model.parameters_dict()
    for key, key_old in key_list:
        key_ = name_raplace(key)
        param = paddle_state_dict[key_old]
        new_param = mindspore.Tensor(param.numpy())
        ms_state_dict[key_].set_data(new_param)
    return ms_state_dict

def pdparams2ckpt(
    pdparams_path: str,
    ms_model,
    save_path: str = '',
):
    """
    paddle参数转mindspore参数并保存ckpt

    :param pdparams_path: paddle参数文件路径
    :param ms_model: 已实例化的mindspore模型
    :param save_path: ckpt保存路径（默认同名）
    :return: None
    """
    if not save_path:
        save_path = pdparams_path.replace('.pdparams', '.ckpt')
    paddle_state = paddle.load(pdparams_path)
    ms_state = update_paddle_to_ms(ms_model, paddle_state)
    mindspore.save_checkpoint(ms_state, save_path)
    print(f'权重已保存到: {save_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='paddle参数转mindspore参数工具')
    parser.add_argument('--pdparams', required=True, help='paddle参数文件路径')
    parser.add_argument('--ms-model-fn', required=True, help='MindSpore模型构造函数（字符串）')
    parser.add_argument('--ckpt', default='', help='ckpt保存路径（可选）')
    parser.add_argument('--ms-args', type=str, default='{}', help='MindSpore模型参数，json格式')
    args = parser.parse_args()
    import json
    ms_args = json.loads(args.ms_args)
    ms_model = eval(args.ms_model_fn)(**ms_args)
    pdparams2ckpt(
        pdparams_path=args.pdparams,
        ms_model=ms_model,
        save_path=args.ckpt
    )
