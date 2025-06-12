"""
通用模型参数转换工具

支持：
- PyTorch(pth) → PaddlePaddle(pdparams)
- PyTorch(pth) → MindSpore(ckpt)
- PaddlePaddle(pdparams) → MindSpore(ckpt)

依赖：torch, paddle, mindspore, numpy, os

用法：
    python model_param_converter.py --config your_config.json

参数说明（config json文件字段）：
- source_type: str，源权重类型（torch/paddle）
- dest_type: str，目标权重类型（paddle/mindspore）
- source_path: str，源权重文件路径
- dest_path: str，目标权重文件路径（可选）
- model_fn: str，目标模型构造函数（字符串，需可eval）
- model_args: dict，目标模型构造参数
- name_change: dict，参数名替换规则（可选）
- filter_list: list，过滤层名列表（可选）

示例配置见 default_model_convert.json
"""
import os
import json

def replace_name_by_rules(key: str, name_change: dict = None) -> str:
    """根据自定义规则批量替换参数名"""
    if name_change:
        for k, v in name_change.items():
            if k in key:
                key = key.replace(k, v)
    return key

def adapt_bn_param_name(key: str, framework: str = 'paddle') -> str:
    """BN等特殊参数名适配，支持不同目标框架"""
    if framework == 'paddle':
        if 'running_mean' in key:
            key = key.replace('running_mean', '_mean')
        if 'running_var' in key:
            key = key.replace('running_var', '_variance')
    elif framework == 'mindspore':
        norm_dict = {'weight': 'gamma', 'bias': 'beta',
                     'running_mean': 'moving_mean', 'running_var': 'moving_variance',
                     '_mean': 'moving_mean', '_variance': 'moving_variance'}
        old = key.split('.')[-1]
        if 'norm' in key or 'bn' in key:
            key = key.replace(old, norm_dict.get(old, old))
        elif ('downsample' in key) and (key.split('.')[-2] != '0'):
            key = key.replace(old, norm_dict.get(old, old))
    return key

def should_filter(key: str, filter_list=None) -> bool:
    """判断参数名是否需要过滤"""
    if not filter_list:
        return False
    return any(f in key for f in filter_list)

def process_param_names(keys, name_change, filter_list, bn_framework):
    """统一参数名处理流程，生成原始和新参数名对"""
    for k in keys:
        if should_filter(k, filter_list):
            continue
        k_new = replace_name_by_rules(k, name_change)
        k_new = adapt_bn_param_name(k_new, framework=bn_framework)
        yield k, k_new

def torch_to_paddle(state_pth, paddle_model, name_change=None, filter_list=None):
    """PyTorch权重转PaddlePaddle权重字典"""
    new_params = paddle_model.state_dict().copy()
    for k, k_new in process_param_names(state_pth.keys(), name_change, filter_list, 'paddle'):
        new_params[k_new] = state_pth[k].cpu().detach().numpy()
    return new_params

def torch_to_mindspore(state_pth, name_change=None, filter_list=None):
    """PyTorch权重转MindSpore权重列表"""
    import mindspore
    static_list_ms = []
    for k, k_new in process_param_names(state_pth.keys(), name_change, filter_list, 'mindspore'):
        param = state_pth[k]
        new_param = mindspore.Parameter(param.detach().numpy())
        static_list_ms.append({'name': k_new, 'data': new_param})
    return static_list_ms

def paddle_to_mindspore(paddle_state, ms_model, name_change=None, filter_list=None):
    """PaddlePaddle权重转MindSpore权重字典"""
    import mindspore
    import numpy as np
    ms_state_dict = ms_model.parameters_dict()
    for k, k_new in process_param_names(paddle_state.keys(), name_change, filter_list, 'mindspore'):
        param = paddle_state[k]
        new_param = mindspore.Tensor(param.numpy())
        if k_new in ms_state_dict:
            ms_state_dict[k_new].set_data(new_param)
    return ms_state_dict

def convert_model_params(
    source_type: str,
    dest_type: str,
    source_path: str,
    dest_path: str = '',
    model_fn: str = '',
    model_args: dict = None,
    name_change: dict = None,
    filter_list: list = None
):
    """
    通用模型参数转换主函数
    :param source_type: 源权重类型（torch/paddle）
    :param dest_type: 目标权重类型（paddle/mindspore）
    :param source_path: 源权重文件路径
    :param dest_path: 目标权重文件路径
    :param model_fn: 目标模型构造函数（字符串，需可eval）
    :param model_args: 目标模型构造参数
    :param name_change: 参数名替换规则
    :param filter_list: 过滤层名列表
    :return: None
    """
    if source_type == 'torch' and dest_type == 'paddle':
        import torch
        import paddle
        if not dest_path:
            dest_path = source_path.replace('.pth', '.pdparams')
        paddle_model = eval(model_fn)(**(model_args or {}))
        state_pth = torch.load(source_path, map_location='cpu')
        new_params = torch_to_paddle(state_pth, paddle_model, name_change, filter_list)
        paddle_model.set_state_dict(new_params)
        paddle.save(new_params, dest_path)
        print(f'权重已保存到: {dest_path}')
    elif source_type == 'torch' and dest_type == 'mindspore':
        import torch
        import mindspore
        if not dest_path:
            dest_path = source_path.replace('.pth', '.ckpt')
        state_pth = torch.load(source_path, map_location='cpu')
        static_list_ms = torch_to_mindspore(state_pth, name_change, filter_list)
        mindspore.save_checkpoint(static_list_ms, dest_path)
        print(f'权重已保存到: {dest_path}')
    elif source_type == 'paddle' and dest_type == 'mindspore':
        import paddle
        import mindspore
        paddle_state = paddle.load(source_path)
        ms_model = eval(model_fn)(**(model_args or {}))
        ms_state_dict = paddle_to_mindspore(paddle_state, ms_model, name_change, filter_list)
        if not dest_path:
            dest_path = source_path.replace('.pdparams', '.ckpt')
        mindspore.save_checkpoint(ms_state_dict, dest_path)
        print(f'权重已保存到: {dest_path}')
    else:
        raise NotImplementedError(f'不支持的转换: {source_type} → {dest_type}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='通用模型参数转换工具')
    parser.add_argument('--config', default='./default_model_convert.json', help='参数配置json文件路径')
    args = parser.parse_args()
    # 读取json配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    # 参数解析
    source_type = config.get('source_type')
    dest_type = config.get('dest_type')
    source_path = config.get('source_path')
    dest_path = config.get('dest_path', '')
    model_fn = config.get('model_fn', '')
    model_args = config.get('model_args', {})
    name_change = config.get('name_change', None)
    filter_list = config.get('filter_list', None)
    convert_model_params(
        source_type=source_type,
        dest_type=dest_type,
        source_path=source_path,
        dest_path=dest_path,
        model_fn=model_fn,
        model_args=model_args,
        name_change=name_change,
        filter_list=filter_list
    )

