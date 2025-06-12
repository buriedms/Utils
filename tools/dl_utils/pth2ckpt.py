"""
PyTorch模型参数转MindSpore模型参数工具

功能：
- PyTorch(pth)参数结构与MindSpore(ckpt)参数结构的自动映射与转换。
- 支持参数名批量替换、BN层参数名适配、过滤特定层。

依赖：mindspore, torch
"""
import mindspore
import torch

def change_param_name(name: str, name_change=None):
    """参数名前缀批量替换"""
    if name_change is None:
        name_change = {}
    if '' in name_change:
        name = name_change[''] + '.' + name
        del name_change['']
    for key in name_change:
        if key in name:
            name = name.replace(key, name_change[key])
            if name.startswith('.'):
                name = name[1:]
    return name

def name_raplace(name):
    """BN层等参数名适配"""
    norm_dict = {'weight': 'gamma', 'bias': 'beta',
                 'running_mean': 'moving_mean', 'running_var': 'moving_variance'}
    old = name.split('.')[-1]
    if 'norm' in name or 'bn' in name:
        return name.replace(old, norm_dict.get(old, old))
    return name

def update_name(list_old, name_change=None, filter_list=None):
    """生成参数名映射及过滤表"""
    if name_change is None:
        name_change = {}
    list_new = [change_param_name(name, name_change=name_change) for name in list_old]
    if not filter_list:
        filter_list = []
    result = []
    for name, name_old in zip(list_new, list_old):
        if any(layer in name for layer in filter_list):
            result.append((name_old, 'delete'))
        else:
            result.append((name_raplace(name), name_old))
    return result

def pth2ckpt(
    pth_path: str,
    ms_save_path: str = '',
    name_change: dict = None,
    filter_list: list = None
):
    """
    PyTorch参数转MindSpore参数主函数

    :param pth_path: pth文件路径
    :param ms_save_path: ckpt保存路径（默认同名）
    :param name_change: 参数名替换规则
    :param filter_list: 过滤层名列表
    :return: MindSpore参数字典
    """
    if not ms_save_path:
        ms_save_path = pth_path.replace('.pth', '.ckpt')
    static_dict_pth = torch.load(pth_path, map_location='cpu')
    key_list = update_name(static_dict_pth.keys(), name_change=name_change, filter_list=filter_list)
    static_list_ms = []
    for key, key_old in key_list:
        if key_old == 'delete':
            continue
        param = static_dict_pth[key_old]
        new_param = mindspore.Parameter(param.detach().numpy())
        static_list_ms.append({'name': key, 'data': new_param})
    mindspore.save_checkpoint(static_list_ms, ms_save_path)
    return {data['name']: data['data'] for data in static_list_ms}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch参数转MindSpore参数工具')
    parser.add_argument('--pth', required=True, help='pth文件路径')
    parser.add_argument('--ckpt', default='', help='ckpt保存路径（可选）')
    parser.add_argument('--name-change', type=str, default='', help='参数名替换规则，格式如 key1:val1,key2:val2')
    parser.add_argument('--filter-list', type=str, default='', help='过滤层名列表，逗号分隔')
    args = parser.parse_args()
    name_change = dict(item.split(':') for item in args.name_change.split(',') if item) if args.name_change else None
    filter_list = args.filter_list.split(',') if args.filter_list else None
    pth2ckpt(
        pth_path=args.pth,
        ms_save_path=args.ckpt,
        name_change=name_change,
        filter_list=filter_list
    )
